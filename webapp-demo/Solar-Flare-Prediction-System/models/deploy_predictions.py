#!/usr/bin/env python3
"""
deploy_predictions.py
Production deployment system for solar flare prediction models.

This script:
1. Runs predictions on all available model combinations (C/M/M5 × 24h/48h/72h)
2. Uses the latest available testing data
3. Generates comprehensive JSON output with all results
4. Provides detailed performance metrics and metadata
5. Can be run periodically to update predictions

Usage:
    python deploy_predictions.py --output-dir ../src/data --update-mode
"""

import os
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, classification_report
)

# Import model and data utilities
from solarknowledge_ret_plus import RETPlusWrapper
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import get_testing_data
from model_tracking import load_model_metadata

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Configuration
INPUT_SHAPE = (10, 9)
THRESHOLD = 0.5
FLARE_CLASSES = ["C", "M", "M5"]
HORIZONS = ["24", "48", "72"]
MODEL_BASE_DIR = "models"

# Version extraction regex
VERSION_PATTERN = re.compile(r"EVEREST-v([\d.]+)-([A-Z0-9]+)-(\d+)h")

class SolarFlarePredictor:
    """Production solar flare prediction system."""
    
    def __init__(self, model_base_dir: str = MODEL_BASE_DIR):
        self.model_base_dir = model_base_dir
        self.results = {}
        self.metadata = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "system_info": {
                "model_base_dir": model_base_dir,
                "threshold": THRESHOLD,
                "input_shape": INPUT_SHAPE
            },
            "models_tested": [],
            "summary": {}
        }
    
    def get_latest_model_path(self, flare_class: str, time_window: str) -> Optional[Tuple[str, str]]:
        """Find the latest model version for a given flare class and time window."""
        candidates = []
        model_dir = os.path.join(self.model_base_dir, "models") if self.model_base_dir != "models" else "models"
        
        if not os.path.exists(model_dir):
            log.warning(f"Model directory {model_dir} not found")
            return None
            
        for dirname in os.listdir(model_dir):
            match = VERSION_PATTERN.fullmatch(dirname)
            if match:
                version, fclass, thours = match.groups()
                if fclass == flare_class and thours == time_window:
                    # Parse version for sorting (handle both x.y and x.y.z formats)
                    version_parts = list(map(int, version.split(".")))
                    candidates.append((version_parts, dirname, version))
        
        if not candidates:
            log.warning(f"No model found for {flare_class}-{time_window}h")
            return None
            
        # Sort by version and get the latest
        latest = sorted(candidates)[-1]
        model_dir_name = latest[1]
        version = latest[2]
        model_path = os.path.join(model_dir, model_dir_name, "model_weights.pt")
        
        if not os.path.exists(model_path):
            log.warning(f"Model weights not found at {model_path}")
            return None
            
        return model_path, version
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
        """Calculate comprehensive performance metrics."""
        # Basic metrics
        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # ROC AUC (if we have probability scores)
        try:
            auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            auc = 0.0
        
        # True Skill Statistic (TSS)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        tss = recall + specificity - 1
        
        # Heidke Skill Score (HSS)
        po = accuracy  # Observed accuracy
        pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (len(y_true) ** 2)
        hss = (po - pe) / (1 - pe) if pe != 1 else 0
        
        return {
            "confusion_matrix": cm.tolist(),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc_roc": float(auc),
            "tss": float(tss),
            "hss": float(hss),
            "specificity": float(specificity),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "total_samples": len(y_true),
            "positive_samples": int(np.sum(y_true)),
            "negative_samples": int(len(y_true) - np.sum(y_true))
        }
    
    def predict_single_model(self, flare_class: str, time_window: str) -> Optional[Dict]:
        """Run predictions for a single model combination."""
        log.info(f"🔍 Testing {flare_class}-class, {time_window}h horizon")
        
        # Find latest model
        model_info = self.get_latest_model_path(flare_class, time_window)
        if not model_info:
            return None
            
        model_path, version = model_info
        log.info(f"Using model: {model_path} (version {version})")
        
        try:
            # Load test data
            X_test, y_test = get_testing_data(time_window, flare_class)
            if X_test is None or y_test is None:
                log.error(f"Failed to load test data for {flare_class}-{time_window}h")
                return None
                
            y_test = np.array(y_test)
            log.info(f"Loaded {len(X_test)} test samples")
            
            # Load model
            model = RETPlusWrapper(INPUT_SHAPE)
            model.load(model_path)
            log.info("Model loaded successfully")
            
            # Make predictions
            y_proba = model.predict_proba(X_test)
            y_pred = (y_proba >= THRESHOLD).astype(int).squeeze()
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred, y_proba.squeeze())
            
            # Load model metadata if available
            try:
                model_metadata = load_model_metadata(version, flare_class, time_window)
            except:
                model_metadata = {}
            
            # Prepare detailed predictions
            predictions_detail = []
            for i in range(len(y_test)):
                predictions_detail.append({
                    "sample_id": i,
                    "true_label": int(y_test[i]),
                    "predicted_label": int(y_pred[i]),
                    "probability": float(y_proba[i] if y_proba.ndim == 1 else y_proba[i, 0]),
                    "correct": bool(y_test[i] == y_pred[i])
                })
            
            result = {
                "model_info": {
                    "flare_class": flare_class,
                    "time_window": int(time_window),
                    "version": version,
                    "model_path": model_path,
                    "threshold": THRESHOLD
                },
                "data_info": {
                    "test_samples": len(X_test),
                    "positive_samples": int(np.sum(y_test)),
                    "negative_samples": int(len(y_test) - np.sum(y_test)),
                    "class_distribution": {
                        "positive_rate": float(np.mean(y_test)),
                        "negative_rate": float(1 - np.mean(y_test))
                    }
                },
                "performance": metrics,
                "model_metadata": model_metadata,
                "predictions": predictions_detail,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            log.info(f"✅ {flare_class}-{time_window}h: Accuracy={metrics['accuracy']:.4f}, TSS={metrics['tss']:.4f}")
            return result
            
        except Exception as e:
            log.error(f"❌ Error processing {flare_class}-{time_window}h: {str(e)}")
            return None
    
    def run_all_predictions(self) -> Dict:
        """Run predictions for all available model combinations."""
        log.info("🚀 Starting comprehensive model evaluation...")
        
        total_models = 0
        successful_models = 0
        
        for flare_class in FLARE_CLASSES:
            for time_window in HORIZONS:
                total_models += 1
                result = self.predict_single_model(flare_class, time_window)
                
                if result:
                    key = f"{flare_class}_{time_window}h"
                    self.results[key] = result
                    self.metadata["models_tested"].append({
                        "flare_class": flare_class,
                        "time_window": time_window,
                        "version": result["model_info"]["version"],
                        "status": "success"
                    })
                    successful_models += 1
                else:
                    self.metadata["models_tested"].append({
                        "flare_class": flare_class,
                        "time_window": time_window,
                        "status": "failed"
                    })
        
        # Generate summary statistics
        self.metadata["summary"] = {
            "total_models_attempted": total_models,
            "successful_models": successful_models,
            "failed_models": total_models - successful_models,
            "success_rate": successful_models / total_models if total_models > 0 else 0
        }
        
        # Calculate aggregate statistics
        if self.results:
            all_accuracies = [r["performance"]["accuracy"] for r in self.results.values()]
            all_tss = [r["performance"]["tss"] for r in self.results.values()]
            
            self.metadata["summary"]["aggregate_stats"] = {
                "mean_accuracy": float(np.mean(all_accuracies)),
                "std_accuracy": float(np.std(all_accuracies)),
                "mean_tss": float(np.mean(all_tss)),
                "std_tss": float(np.std(all_tss)),
                "best_accuracy": float(np.max(all_accuracies)),
                "best_tss": float(np.max(all_tss))
            }
        
        log.info(f"🎯 Completed: {successful_models}/{total_models} models successful")
        return {
            "metadata": self.metadata,
            "results": self.results
        }
    
    def save_results(self, output_path: str, results: Dict):
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        log.info(f"💾 Results saved to {output_path}")
        
        # Also save a compact version for web UI
        compact_path = output_path.replace('.json', '_compact.json')
        compact_results = {
            "metadata": results["metadata"],
            "summary": {
                key: {
                    "model_info": value["model_info"],
                    "performance": value["performance"],
                    "data_info": value["data_info"]
                }
                for key, value in results["results"].items()
            }
        }
        
        with open(compact_path, 'w') as f:
            json.dump(compact_results, f, indent=2, default=str)
        
        log.info(f"💾 Compact results saved to {compact_path}")

def main():
    parser = argparse.ArgumentParser(description="Deploy solar flare prediction models")
    parser.add_argument("--output-dir", type=str, default="../../src/data",
                       help="Output directory for prediction results")
    parser.add_argument("--model-dir", type=str, default=MODEL_BASE_DIR,
                       help="Base directory containing models")
    parser.add_argument("--update-mode", action="store_true",
                       help="Run in update mode (for periodic updates)")
    parser.add_argument("--compact-only", action="store_true",
                       help="Only generate compact output (for web UI)")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = SolarFlarePredictor(args.model_dir)
    
    # Run predictions
    results = predictor.run_all_predictions()
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.update_mode:
        output_file = os.path.join(args.output_dir, "latest_predictions.json")
    else:
        output_file = os.path.join(args.output_dir, f"predictions_{timestamp}.json")
    
    # Save results
    predictor.save_results(output_file, results)
    
    # Print summary
    print("\n" + "="*60)
    print("🌟 SOLAR FLARE PREDICTION DEPLOYMENT SUMMARY")
    print("="*60)
    print(f"📊 Models tested: {results['metadata']['summary']['successful_models']}/{results['metadata']['summary']['total_models_attempted']}")
    
    if results['results']:
        print(f"📈 Mean accuracy: {results['metadata']['summary']['aggregate_stats']['mean_accuracy']:.4f}")
        print(f"📈 Mean TSS: {results['metadata']['summary']['aggregate_stats']['mean_tss']:.4f}")
        
        print("\n🏆 Best performing models:")
        sorted_results = sorted(
            results['results'].items(), 
            key=lambda x: x[1]['performance']['tss'], 
            reverse=True
        )
        
        for i, (key, result) in enumerate(sorted_results[:3]):
            print(f"  {i+1}. {key}: TSS={result['performance']['tss']:.4f}, Acc={result['performance']['accuracy']:.4f}")
    
    print(f"\n💾 Results saved to: {output_file}")
    print("="*60)

if __name__ == "__main__":
    main() 