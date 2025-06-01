#!/usr/bin/env python3
"""
CALCULATE BRIER SCORES FOR PAPER
Calculate 99th percentile Brier scores for SolarKnowledge vs EVEREST
to get actual values for the paper section about high-confidence forecast calibration.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from pathlib import Path
from sklearn.metrics import brier_score_loss
import warnings
warnings.filterwarnings('ignore')

# Add models to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "models"))

def load_test_data():
    """Load SHARP M5-72h test data."""
    print("Loading SHARP M5-72h test data...")
    test_data_path = project_root / "Nature_data/testing_data_M5_72.csv"
    df = pd.read_csv(test_data_path)
    
    # Filter out padding rows
    df = df[df['Flare'] != 'padding'].copy()
    
    # Extract features and labels
    feature_columns = ['USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANPOT', 
                      'TOTUSJH', 'TOTPOT', 'ABSNJZH', 'SAVNCPP']
    
    X_test = df[feature_columns].values
    y_test = (df['Flare'] == 'P').astype(int).values
    
    # Reshape for models: (samples, timesteps=10, features=9)
    n_samples = len(X_test) // 10
    X_test = X_test[:n_samples*10].reshape(n_samples, 10, 9)
    y_test = y_test[:n_samples*10:10]  # Take every 10th label
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"Positive class samples: {y_test.sum()}/{len(y_test)} ({100*y_test.sum()/len(y_test):.2f}%)")
    
    return X_test, y_test

def load_solarknowledge_model():
    """Load retrained SolarKnowledge model."""
    print("Loading SolarKnowledge model...")
    
    model_path = project_root / "models/models/SolarKnowledge retrained/Solarknowledge-v4.5_10582_20250531_070617_716154-M5-24h"
    weight_path = model_path / "model_weights.weights.h5"
    
    from SolarKnowledge_model import SolarKnowledge
    
    model = SolarKnowledge(early_stopping_patience=5)
    model.build_base_model(
        input_shape=(10, 9),
        embed_dim=128,
        num_heads=4,
        ff_dim=256,
        num_transformer_blocks=6,
        dropout_rate=0.2,
        num_classes=2
    )
    model.compile(
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        learning_rate=0.0001,
        use_focal_loss=True
    )
    model.model.load_weights(str(weight_path))
    print("✅ SolarKnowledge model loaded")
    
    return model

def load_everest_model():
    """Load EVEREST model."""
    print("Loading EVEREST model...")
    
    everest_weights_path = project_root / "tests/model_weights_EVEREST_72h_M5.pt"
    
    from solarknowledge_ret_plus import RETPlusWrapper
    
    # Force CPU device for compatibility
    device = torch.device('cpu')
    
    model = RETPlusWrapper(input_shape=(10, 9))
    
    # Build the model with CPU device
    dummy_input = torch.randn(1, 10, 9, device=device)
    _ = model.model(dummy_input)
    
    # Load weights
    checkpoint = torch.load(everest_weights_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.model.load_state_dict(checkpoint)
    
    # Ensure model is on CPU
    model.model.to(device)
    model.model.eval()
    print("✅ EVEREST model loaded (CPU)")
    
    return model

def get_solarknowledge_predictions(model, X_test):
    """Get SolarKnowledge predictions."""
    print("Getting SolarKnowledge predictions...")
    
    y_pred_probs = model.predict(X_test, batch_size=512, verbose=0)
    
    # Handle categorical output
    if len(y_pred_probs.shape) > 1 and y_pred_probs.shape[1] > 1:
        y_pred_probs_pos = y_pred_probs[:, 1]  # Take positive class probability
    else:
        y_pred_probs_pos = y_pred_probs.flatten()
    
    return y_pred_probs_pos

def get_everest_predictions(model, X_test):
    """Get EVEREST predictions."""
    print("Getting EVEREST predictions...")
    
    # Ensure CPU tensors
    device = torch.device('cpu')
    X_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        model.model.eval()
        outputs = model.model(X_tensor)
        
        # Extract probabilities from composite output
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            y_pred_logits = outputs[0]
        else:
            y_pred_logits = outputs
        
        # Apply sigmoid to get probabilities
        y_pred_probs = torch.sigmoid(y_pred_logits).squeeze().cpu().numpy()
    
    return y_pred_probs

def calculate_brier_score_percentiles(y_true, y_pred_probs):
    """Calculate Brier scores and percentiles."""
    
    # Calculate individual Brier scores for each prediction
    individual_brier_scores = (y_pred_probs - y_true) ** 2
    
    # Calculate overall Brier score
    overall_brier = np.mean(individual_brier_scores)
    
    # Calculate percentiles
    percentiles = [50, 90, 95, 99, 99.9]
    brier_percentiles = {}
    
    for p in percentiles:
        percentile_value = np.percentile(individual_brier_scores, p)
        brier_percentiles[p] = percentile_value
    
    return {
        'overall_brier': overall_brier,
        'individual_scores': individual_brier_scores,
        'percentiles': brier_percentiles
    }

def analyze_high_confidence_calibration(y_true, y_pred_probs, confidence_threshold=0.9):
    """Analyze calibration performance for high-confidence predictions."""
    
    high_conf_mask = y_pred_probs >= confidence_threshold
    
    if high_conf_mask.sum() > 0:
        high_conf_probs = y_pred_probs[high_conf_mask]
        high_conf_true = y_true[high_conf_mask]
        
        # Brier score for high-confidence predictions
        high_conf_brier = np.mean((high_conf_probs - high_conf_true) ** 2)
        
        # Calibration metrics
        mean_confidence = np.mean(high_conf_probs)
        mean_accuracy = np.mean(high_conf_true)
        calibration_gap = abs(mean_confidence - mean_accuracy)
        
        return {
            'count': high_conf_mask.sum(),
            'percentage': 100 * high_conf_mask.sum() / len(y_true),
            'brier_score': high_conf_brier,
            'mean_confidence': mean_confidence,
            'mean_accuracy': mean_accuracy,
            'calibration_gap': calibration_gap
        }
    else:
        return None

def print_brier_comparison(sk_results, ev_results):
    """Print comprehensive Brier score comparison."""
    
    print("\n" + "="*80)
    print("BRIER SCORE ANALYSIS FOR PAPER")
    print("="*80)
    
    print(f"\n📊 OVERALL BRIER SCORES:")
    print(f"   SolarKnowledge: {sk_results['overall_brier']:.6f}")
    print(f"   EVEREST:        {ev_results['overall_brier']:.6f}")
    
    overall_improvement = ((sk_results['overall_brier'] - ev_results['overall_brier']) / sk_results['overall_brier']) * 100
    print(f"   Improvement:    {overall_improvement:.1f}%")
    
    print(f"\n📈 BRIER SCORE PERCENTILES:")
    print("   Percentile    SolarKnowledge    EVEREST       Improvement")
    print("   " + "-"*60)
    
    for p in [50, 90, 95, 99, 99.9]:
        sk_val = sk_results['percentiles'][p]
        ev_val = ev_results['percentiles'][p]
        improvement = ((sk_val - ev_val) / sk_val) * 100 if sk_val > 0 else 0
        
        print(f"   {p:>6.1f}%        {sk_val:>8.6f}      {ev_val:>8.6f}     {improvement:>7.1f}%")
    
    # Specific focus on 99th percentile for paper
    sk_99th = sk_results['percentiles'][99]
    ev_99th = ev_results['percentiles'][99]
    improvement_99th = ((sk_99th - ev_99th) / sk_99th) * 100
    
    print(f"\n🎯 99TH PERCENTILE FOCUS (FOR PAPER):")
    print(f"   SolarKnowledge 99th percentile Brier: {sk_99th:.6f}")
    print(f"   EVEREST 99th percentile Brier:        {ev_99th:.6f}")
    print(f"   Reduction:                             {improvement_99th:.1f}%")
    print(f"   Paper text: \"{improvement_99th:.0f}% reduction in 99th-percentile Brier score\"")
    print(f"               \"(from {sk_99th:.3f} to {ev_99th:.4f})\"")

def main():
    """Run Brier score analysis."""
    print("🧪 BRIER SCORE ANALYSIS FOR PAPER SECTION")
    print("="*60)
    print("Calculating 99th percentile Brier scores for SolarKnowledge vs EVEREST")
    
    try:
        # Load test data
        X_test, y_test = load_test_data()
        
        # Load models
        sk_model = load_solarknowledge_model()
        ev_model = load_everest_model()
        
        # Get predictions
        sk_probs = get_solarknowledge_predictions(sk_model, X_test)
        ev_probs = get_everest_predictions(ev_model, X_test)
        
        # Calculate Brier score analysis
        print("\nCalculating Brier score percentiles...")
        sk_results = calculate_brier_score_percentiles(y_test, sk_probs)
        ev_results = calculate_brier_score_percentiles(y_test, ev_probs)
        
        # Print comparison
        print_brier_comparison(sk_results, ev_results)
        
        # High-confidence analysis
        print(f"\n🔍 HIGH-CONFIDENCE PREDICTION ANALYSIS:")
        sk_high_conf = analyze_high_confidence_calibration(y_test, sk_probs, 0.9)
        ev_high_conf = analyze_high_confidence_calibration(y_test, ev_probs, 0.9)
        
        if sk_high_conf:
            print(f"   SolarKnowledge (p≥0.9): {sk_high_conf['count']} predictions, Brier = {sk_high_conf['brier_score']:.6f}")
        
        if ev_high_conf:
            print(f"   EVEREST (p≥0.9):        {ev_high_conf['count']} predictions, Brier = {ev_high_conf['brier_score']:.6f}")
        
        # Save results for paper
        results_file = project_root / "brier_score_analysis.txt"
        with open(results_file, 'w') as f:
            f.write("BRIER SCORE ANALYSIS FOR PAPER\n")
            f.write("="*40 + "\n\n")
            
            sk_99th = sk_results['percentiles'][99]
            ev_99th = ev_results['percentiles'][99]
            improvement_99th = ((sk_99th - ev_99th) / sk_99th) * 100
            
            f.write("99TH PERCENTILE BRIER SCORE RESULTS:\n")
            f.write(f"SolarKnowledge: {sk_99th:.6f}\n")
            f.write(f"EVEREST: {ev_99th:.6f}\n")
            f.write(f"Improvement: {improvement_99th:.1f}%\n\n")
            
            f.write("PAPER TEXT:\n")
            f.write(f"\\textbf{{{improvement_99th:.0f}\\% reduction}} in the 99$^{{\\text{{th}}}}$-percentile Brier score\n")
            f.write(f"(from {sk_99th:.3f} to {ev_99th:.4f}), confirming its effectiveness in calibrating\n")
            f.write("high-confidence forecasts.\n")
        
        print(f"\n💾 Results saved to: {results_file}")
        
        return {
            'sk_results': sk_results,
            'ev_results': ev_results,
            'sk_99th': sk_99th,
            'ev_99th': ev_99th,
            'improvement_99th': improvement_99th
        }
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main() 