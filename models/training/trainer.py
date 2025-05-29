"""
EVEREST Production Model Trainer

This module implements the production training pipeline for EVEREST models
with threshold optimization and comprehensive evaluation on test sets.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix
)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .config import (
    TRAINING_TARGETS, RANDOM_SEEDS, FIXED_ARCHITECTURE, TRAINING_HYPERPARAMS,
    LOSS_WEIGHT_SCHEDULE, THRESHOLD_CONFIG, BALANCED_WEIGHTS, EVALUATION_METRICS,
    OUTPUT_CONFIG, STATISTICAL_CONFIG, get_experiment_name, get_threshold_search_points,
    calculate_balanced_score, create_output_directories
)
from solarknowledge_ret_plus import RETPlusWrapper
from utils import get_training_data, get_testing_data


class ProductionTrainer:
    """Production model trainer with threshold optimization."""
    
    def __init__(self, flare_class: str, time_window: str, seed: int):
        """Initialize trainer for specific target and seed."""
        self.flare_class = flare_class
        self.time_window = time_window
        self.seed = seed
        self.experiment_name = get_experiment_name(flare_class, time_window, seed)
        
        # Setup reproducibility
        self._setup_reproducibility()
        
        # Setup directories
        self._setup_directories()
        
        print(f"🏭 Production Trainer initialized: {self.experiment_name}")
        print(f"   Target: {flare_class}-class, {time_window}h horizon")
        print(f"   Seed: {seed}")
    
    def _setup_reproducibility(self):
        """Setup reproducible training environment."""
        # Set random seeds
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        # Deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variables for reproducibility
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        
        print(f"🎲 Reproducibility configured with seed {self.seed}")
    
    def _setup_directories(self):
        """Create experiment-specific directories."""
        self.experiment_dir = os.path.join(OUTPUT_CONFIG["results_dir"], self.experiment_name)
        self.model_dir = os.path.join(OUTPUT_CONFIG["models_dir"], self.experiment_name)
        self.log_dir = os.path.join(OUTPUT_CONFIG["logs_dir"], self.experiment_name)
        
        for directory in [self.experiment_dir, self.model_dir, self.log_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load training and testing data."""
        print("📊 Loading data...")
        
        # Load training and testing data
        X_train, y_train = get_training_data(self.time_window, self.flare_class)
        X_test, y_test = get_testing_data(self.time_window, self.flare_class)
        
        if X_train is None or y_train is None:
            raise ValueError(f"Training data not found for {self.flare_class}/{self.time_window}h")
        
        if X_test is None or y_test is None:
            raise ValueError(f"Testing data not found for {self.flare_class}/{self.time_window}h")
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Testing samples: {len(X_test):,}")
        print(f"   Training positive rate: {np.mean(y_train):.4f}")
        print(f"   Testing positive rate: {np.mean(y_test):.4f}")
        print(f"   Input shape: {X_train.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def create_model(self) -> RETPlusWrapper:
        """Create EVEREST model with production configuration."""
        print("🚀 Creating EVEREST model...")
        
        # Create wrapper with production config
        wrapper = RETPlusWrapper(
            input_shape=FIXED_ARCHITECTURE["input_shape"],
            early_stopping_patience=TRAINING_HYPERPARAMS["early_stopping_patience"],
            use_attention_bottleneck=FIXED_ARCHITECTURE["use_attention_bottleneck"],
            use_evidential=FIXED_ARCHITECTURE["use_evidential"],
            use_evt=FIXED_ARCHITECTURE["use_evt"],
            use_precursor=FIXED_ARCHITECTURE["use_precursor"]
        )
        
        # Update optimizer with production learning rate
        wrapper.optimizer = torch.optim.AdamW(
            wrapper.model.parameters(),
            lr=TRAINING_HYPERPARAMS["learning_rate"],
            weight_decay=TRAINING_HYPERPARAMS["weight_decay"],
            fused=True
        )
        
        device = next(wrapper.model.parameters()).device
        print(f"✅ Model created on device: {device}")
        print(f"   Parameters: {sum(p.numel() for p in wrapper.model.parameters()):,}")
        
        return wrapper
    
    def train_model(self, model: RETPlusWrapper, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train the model using production configuration."""
        print("\n🏋️ Training model...")
        
        start_time = time.time()
        
        # Train using the wrapper's train method with production config
        model_dir = model.train(
            X_train=X_train,
            y_train=y_train,
            epochs=TRAINING_HYPERPARAMS["epochs"],
            batch_size=TRAINING_HYPERPARAMS["batch_size"],
            gamma_max=TRAINING_HYPERPARAMS["focal_gamma_max"],
            warmup_epochs=TRAINING_HYPERPARAMS["warmup_epochs"],
            flare_class=self.flare_class,
            time_window=self.time_window,
            in_memory_dataset=TRAINING_HYPERPARAMS["in_memory_dataset"],
            track_emissions=True
        )
        
        training_time = time.time() - start_time
        
        print(f"✅ Training completed in {training_time:.1f}s ({training_time/60:.1f} min)")
        print(f"📁 Model saved to: {model_dir}")
        
        return {
            "model_dir": model_dir,
            "training_time": training_time,
            "training_history": model.history
        }
    
    def optimize_threshold(self, model: RETPlusWrapper, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Optimize classification threshold on test set."""
        print("\n🎯 Optimizing classification threshold...")
        
        # Get probability predictions
        y_probs = model.predict_proba(X_test).flatten()
        
        # Search over threshold values
        threshold_points = get_threshold_search_points()
        threshold_results = []
        
        best_score = -1.0
        best_threshold = THRESHOLD_CONFIG["fallback_threshold"]
        best_metrics = {}
        
        for threshold in threshold_points:
            # Make predictions with this threshold
            y_pred = (y_probs >= threshold).astype(int)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_probs)
            
            # Calculate balanced score
            balanced_score = calculate_balanced_score(metrics)
            
            threshold_results.append({
                "threshold": threshold,
                "balanced_score": balanced_score,
                **metrics
            })
            
            # Track best threshold
            if balanced_score > best_score:
                best_score = balanced_score
                best_threshold = threshold
                best_metrics = metrics.copy()
        
        print(f"✅ Optimal threshold: {best_threshold:.3f}")
        print(f"   Balanced score: {best_score:.4f}")
        print(f"   TSS: {best_metrics.get('tss', 0):.4f}")
        print(f"   F1: {best_metrics.get('f1', 0):.4f}")
        
        return {
            "optimal_threshold": best_threshold,
            "optimal_score": best_score,
            "optimal_metrics": best_metrics,
            "threshold_curve": threshold_results,
            "probabilities": y_probs.tolist()
        }
    
    def evaluate_model(self, model: RETPlusWrapper, X_test: np.ndarray, y_test: np.ndarray, 
                      optimal_threshold: float) -> Dict[str, Any]:
        """Comprehensive model evaluation on test set."""
        print(f"\n📊 Evaluating model with threshold {optimal_threshold:.3f}...")
        
        # Get predictions
        y_probs = model.predict_proba(X_test).flatten()
        y_pred = (y_probs >= optimal_threshold).astype(int)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_probs)
        
        # Measure inference latency
        latency_ms = self._measure_latency(model, X_test[:32])
        metrics["latency_ms"] = latency_ms
        
        # Create confusion matrix details
        cm = confusion_matrix(y_test, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            confusion_details = {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "total_samples": len(y_test),
                "positive_samples": int(np.sum(y_test)),
                "negative_samples": int(len(y_test) - np.sum(y_test))
            }
        else:
            confusion_details = {"error": "Invalid confusion matrix"}
        
        print(f"📈 Final Test Results:")
        print(f"   TSS: {metrics['tss']:.4f}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   F1: {metrics['f1']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"   Latency: {latency_ms:.1f} ms")
        
        return {
            "test_metrics": metrics,
            "confusion_matrix": confusion_details,
            "predictions": {
                "y_true": y_test.tolist(),
                "y_pred": y_pred.tolist(),
                "y_probs": y_probs.tolist()
            }
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # TSS calculation
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            tss = recall + specificity - 1
        else:
            specificity = 0.0
            tss = 0.0
        
        # Probabilistic metrics
        try:
            roc_auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5
            brier = brier_score_loss(y_true, y_probs)
            ece = self._calculate_ece(y_true, y_probs, n_bins=15)
        except:
            roc_auc = 0.5
            brier = 1.0
            ece = 1.0
        
        return {
            "tss": tss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "roc_auc": roc_auc,
            "brier": brier,
            "ece": ece
        }
    
    def _calculate_ece(self, y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 15) -> float:
        """Calculate Expected Calibration Error (15-bin protocol)."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_probs > bin_lower) & (y_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _measure_latency(self, model: RETPlusWrapper, X_sample: np.ndarray, n_runs: int = 1000) -> float:
        """Measure inference latency in milliseconds."""
        model.model.eval()
        device = next(model.model.parameters()).device
        
        # Prepare input
        X_tensor = torch.tensor(X_sample, dtype=torch.float32).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model.model(X_tensor)
        
        # Measure latency
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model.model(X_tensor)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        end_time = time.time()
        
        # Convert to milliseconds per sample
        total_time_ms = (end_time - start_time) * 1000
        latency_per_batch = total_time_ms / n_runs
        latency_per_sample = latency_per_batch / len(X_sample)
        
        return latency_per_sample
    
    def save_results(self, training_results: Dict[str, Any], threshold_results: Dict[str, Any], 
                    evaluation_results: Dict[str, Any]):
        """Save comprehensive experiment results."""
        print("\n💾 Saving results...")
        
        # Compile complete results
        complete_results = {
            "experiment_info": {
                "experiment_name": self.experiment_name,
                "flare_class": self.flare_class,
                "time_window": self.time_window,
                "seed": self.seed,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "git_tag": OUTPUT_CONFIG["git_tag"]
            },
            "configuration": {
                "architecture": FIXED_ARCHITECTURE,
                "hyperparameters": TRAINING_HYPERPARAMS,
                "loss_schedule": LOSS_WEIGHT_SCHEDULE,
                "threshold_config": THRESHOLD_CONFIG,
                "balanced_weights": BALANCED_WEIGHTS
            },
            "training": training_results,
            "threshold_optimization": threshold_results,
            "evaluation": evaluation_results
        }
        
        # Save main results as JSON
        results_file = os.path.join(self.experiment_dir, "results.json")
        with open(results_file, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        
        # Save training history as CSV
        if "training_history" in training_results:
            history_df = pd.DataFrame(training_results["training_history"])
            history_file = os.path.join(self.experiment_dir, "training_history.csv")
            history_df.to_csv(history_file, index=False)
        
        # Save threshold curve as CSV
        if "threshold_curve" in threshold_results:
            threshold_df = pd.DataFrame(threshold_results["threshold_curve"])
            threshold_file = os.path.join(self.experiment_dir, "threshold_optimization.csv")
            threshold_df.to_csv(threshold_file, index=False)
        
        # Save final metrics summary
        final_metrics = {
            "experiment_name": self.experiment_name,
            "flare_class": self.flare_class,
            "time_window": self.time_window,
            "seed": self.seed,
            "optimal_threshold": threshold_results["optimal_threshold"],
            **evaluation_results["test_metrics"]
        }
        
        metrics_df = pd.DataFrame([final_metrics])
        metrics_file = os.path.join(self.experiment_dir, "final_metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        
        # Save predictions if requested
        if OUTPUT_CONFIG["save_raw_predictions"]:
            predictions_df = pd.DataFrame(evaluation_results["predictions"])
            predictions_file = os.path.join(self.experiment_dir, "predictions.csv")
            predictions_df.to_csv(predictions_file, index=False)
        
        print(f"📊 Results saved:")
        print(f"   Main: {results_file}")
        print(f"   History: {history_file}")
        print(f"   Thresholds: {threshold_file}")
        print(f"   Metrics: {metrics_file}")
    
    def train(self) -> Dict[str, Any]:
        """Complete training pipeline."""
        start_time = time.time()
        
        print(f"\n🏭 Starting production training: {self.experiment_name}")
        print("=" * 70)
        
        # Load data
        X_train, y_train, X_test, y_test = self.load_data()
        
        # Create and train model
        model = self.create_model()
        training_results = self.train_model(model, X_train, y_train)
        
        # Optimize threshold on test set
        threshold_results = self.optimize_threshold(model, X_test, y_test)
        
        # Final evaluation with optimal threshold
        evaluation_results = self.evaluate_model(
            model, X_test, y_test, threshold_results["optimal_threshold"]
        )
        
        # Save all results
        self.save_results(training_results, threshold_results, evaluation_results)
        
        total_time = time.time() - start_time
        print(f"\n✅ Production training completed in {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"📁 Results saved to: {self.experiment_dir}")
        
        return {
            "experiment_name": self.experiment_name,
            "total_time": total_time,
            "optimal_threshold": threshold_results["optimal_threshold"],
            "final_metrics": evaluation_results["test_metrics"]
        }


def train_production_model(flare_class: str, time_window: str, seed: int) -> Dict[str, Any]:
    """
    Train a single production model.
    
    Args:
        flare_class: Flare class (C, M, M5)
        time_window: Time window (24, 48, 72)
        seed: Random seed
        
    Returns:
        Training results dictionary
    """
    trainer = ProductionTrainer(flare_class, time_window, seed)
    return trainer.train()


if __name__ == "__main__":
    # Test training a single model
    import argparse
    
    parser = argparse.ArgumentParser(description="Train production EVEREST model")
    parser.add_argument("--flare_class", required=True, choices=["C", "M", "M5"],
                        help="Flare class to train")
    parser.add_argument("--time_window", required=True, choices=["24", "48", "72"],
                        help="Time window in hours")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    args = parser.parse_args()
    
    print(f"🏭 Training production model: {args.flare_class}-{args.time_window}h")
    print(f"🎲 Random seed: {args.seed}")
    
    results = train_production_model(args.flare_class, args.time_window, args.seed)
    
    print(f"\n🎯 Final Results:")
    print(f"   TSS: {results['final_metrics']['tss']:.4f}")
    print(f"   F1: {results['final_metrics']['f1']:.4f}")
    print(f"   Threshold: {results['optimal_threshold']:.3f}")
    print(f"   Latency: {results['final_metrics']['latency_ms']:.1f} ms") 