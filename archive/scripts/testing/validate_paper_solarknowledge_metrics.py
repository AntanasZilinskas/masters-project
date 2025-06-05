#!/usr/bin/env python3
"""
VALIDATE PAPER SOLARKNOWLEDGE METRICS
Test SolarKnowledge to validate the specific metrics claimed in the paper:
- ECE = 0.185 (mentioned in EVEREST comparison)
- Over-confidence at p ≳ 0.83 (89% confidence, 50% accuracy)
- Precision struggles (without specific values)
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, balanced_accuracy_score, confusion_matrix)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add models to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "models"))

def load_test_data():
    """Load SHARP M5-72h test data (as referenced in paper)."""
    print("Loading SHARP M5-72h test data for paper validation...")
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

def load_retrained_solarknowledge():
    """Load the retrained SolarKnowledge M5-24h model (best available)."""
    print("Loading retrained SolarKnowledge model for paper validation...")
    
    # Use the retrained M5-24h model (closest to paper configuration)
    model_path = project_root / "models/models/SolarKnowledge retrained/Solarknowledge-v4.5_10582_20250531_070617_716154-M5-24h"
    weight_path = model_path / "model_weights.weights.h5"
    
    if not weight_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {weight_path}")
    
    from SolarKnowledge_model import SolarKnowledge
    
    # Initialize model
    model = SolarKnowledge(early_stopping_patience=5)
    
    # Build model with exact parameters from metadata
    model.build_base_model(
        input_shape=(10, 9),
        embed_dim=128,
        num_heads=4,
        ff_dim=256,
        num_transformer_blocks=6,
        dropout_rate=0.2,
        num_classes=2
    )
    
    # Compile model
    model.compile(
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        learning_rate=0.0001,
        use_focal_loss=True
    )
    
    # Load weights
    model.model.load_weights(str(weight_path))
    print("✅ SolarKnowledge model loaded successfully!")
    print(f"Model parameters: {model.model.count_params():,}")
    
    return model

def calculate_detailed_ece(y_true, y_pred_probs, n_bins=15):
    """Calculate ECE with detailed bin analysis for paper validation."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]
    
    ece = 0
    bin_details = []
    
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Find predictions in this bin
        in_bin = (y_pred_probs > bin_lower) & (y_pred_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Calculate accuracy and confidence for this bin
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred_probs[in_bin].mean()
            
            # Add to ECE
            bin_ece = np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            ece += bin_ece
            
            bin_details.append({
                'bin': i,
                'range': f"({bin_lower:.2f}, {bin_upper:.2f}]",
                'count': in_bin.sum(),
                'prop': prop_in_bin,
                'avg_confidence': avg_confidence_in_bin,
                'accuracy': accuracy_in_bin,
                'gap': avg_confidence_in_bin - accuracy_in_bin,
                'contribution': bin_ece
            })
    
    return ece, bin_details

def analyze_overconfidence_threshold(y_true, y_pred_probs, threshold=0.83):
    """Analyze over-confidence at specific threshold (paper claims p ≳ 0.83)."""
    high_conf_mask = y_pred_probs > threshold
    
    if high_conf_mask.sum() > 0:
        high_conf_accuracy = y_true[high_conf_mask].mean()
        high_conf_mean_prob = y_pred_probs[high_conf_mask].mean()
        n_high_conf = high_conf_mask.sum()
        
        return {
            'threshold': threshold,
            'count': n_high_conf,
            'percentage': 100 * n_high_conf / len(y_true),
            'mean_confidence': high_conf_mean_prob,
            'actual_accuracy': high_conf_accuracy,
            'overconfidence_gap': high_conf_mean_prob - high_conf_accuracy
        }
    else:
        return None

def validate_paper_metrics(model, X_test, y_test):
    """Validate specific metrics mentioned in the paper."""
    print("Validating paper-claimed SolarKnowledge metrics...")
    
    # Get predictions
    y_pred_probs = model.predict(X_test, batch_size=512, verbose=0)
    
    # Handle categorical output
    if len(y_pred_probs.shape) > 1 and y_pred_probs.shape[1] > 1:
        y_pred_probs_pos = y_pred_probs[:, 1]  # Take positive class probability
    else:
        y_pred_probs_pos = y_pred_probs.flatten()
    
    # Binary predictions
    y_pred = (y_pred_probs_pos > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Detailed ECE analysis
    ece, bin_details = calculate_detailed_ece(y_test, y_pred_probs_pos)
    
    # Over-confidence analysis at p ≳ 0.83 (paper claim)
    overconf_083 = analyze_overconfidence_threshold(y_test, y_pred_probs_pos, 0.83)
    
    # Check other thresholds for comparison
    overconf_080 = analyze_overconfidence_threshold(y_test, y_pred_probs_pos, 0.80)
    overconf_090 = analyze_overconfidence_threshold(y_test, y_pred_probs_pos, 0.90)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'ece': ece,
        'bin_details': bin_details,
        'overconf_083': overconf_083,
        'overconf_080': overconf_080,
        'overconf_090': overconf_090,
        'probabilities': y_pred_probs_pos,
        'predictions': y_pred,
        'n_samples': len(y_test),
        'n_positives': y_test.sum()
    }

def print_paper_validation_results(results):
    """Print validation results comparing to paper claims."""
    print("\n" + "="*80)
    print("PAPER SOLARKNOWLEDGE METRIC VALIDATION")
    print("="*80)
    
    print("\n🎯 PAPER CLAIMS vs MEASURED RESULTS:")
    print(f"   Paper ECE Claim:      0.185 (SolarKnowledge → EVEREST comparison)")
    print(f"   Measured ECE:         {results['ece']:.6f}")
    
    if abs(results['ece'] - 0.185) < 0.05:
        print("   ✅ ECE matches paper claim (within 0.05)")
    else:
        print("   ⚠️  ECE differs from paper claim")
    
    print(f"\n📊 CLASSIFICATION METRICS:")
    print(f"   Accuracy:    {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"   Precision:   {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"   Recall:      {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"   F1-Score:    {results['f1_score']:.4f}")
    
    # Precision analysis (paper mentions "struggles with precision")
    if results['precision'] < 0.5:
        print("   ✅ Confirms precision struggles mentioned in paper")
    else:
        print("   ⚠️  Precision better than expected")
    
    print(f"\n🔍 OVER-CONFIDENCE ANALYSIS:")
    print(f"   Paper Claim: p ≳ 0.83 shows 89% confidence, 50% accuracy")
    
    if results['overconf_083']:
        oc = results['overconf_083']
        print(f"   Measured at p > 0.83:")
        print(f"     • Count: {oc['count']} samples ({oc['percentage']:.1f}%)")
        print(f"     • Mean Confidence: {oc['mean_confidence']:.1%}")
        print(f"     • Actual Accuracy: {oc['actual_accuracy']:.1%}")
        print(f"     • Over-confidence Gap: {oc['overconfidence_gap']:.3f}")
        
        # Check if matches paper claim
        if abs(oc['mean_confidence'] - 0.89) < 0.1 and abs(oc['actual_accuracy'] - 0.50) < 0.2:
            print("   ✅ Matches paper over-confidence claim")
        else:
            print("   ⚠️  Differs from paper over-confidence claim")
    else:
        print("   ❌ No samples with p > 0.83 found")
    
    print(f"\n📈 CALIBRATION BIN ANALYSIS:")
    print("   Bin Range        Count  Confidence  Accuracy   Gap     ECE Contrib.")
    print("   " + "-"*65)
    
    for bin_detail in results['bin_details']:
        if bin_detail['count'] > 0:
            print(f"   {bin_detail['range']:12} {bin_detail['count']:6} "
                  f"{bin_detail['avg_confidence']:10.3f} {bin_detail['accuracy']:9.3f} "
                  f"{bin_detail['gap']:7.3f} {bin_detail['contribution']:11.6f}")
    
    print(f"\n💡 PAPER VALIDATION SUMMARY:")
    if results['ece'] > 0.15:
        print("   ✅ High ECE confirms calibration issues mentioned in paper")
    if results['precision'] < 0.3:
        print("   ✅ Low precision confirms precision struggles mentioned in paper")
    if results['overconf_083'] and results['overconf_083']['overconfidence_gap'] > 0.2:
        print("   ✅ Over-confidence gap confirms reliability issues mentioned in paper")

def create_reliability_plot(results):
    """Create reliability plot similar to paper Figure."""
    bin_details = results['bin_details']
    
    if not bin_details:
        print("No bin details available for plotting")
        return
    
    # Extract data for plotting
    bin_centers = []
    accuracies = []
    confidences = []
    counts = []
    
    for bd in bin_details:
        if bd['count'] > 0:
            bin_center = (float(bd['range'].split('(')[1].split(',')[0]) + 
                         float(bd['range'].split(', ')[1].split(']')[0])) / 2
            bin_centers.append(bin_center)
            accuracies.append(bd['accuracy'])
            confidences.append(bd['avg_confidence'])
            counts.append(bd['count'])
    
    if not bin_centers:
        print("No valid bins for plotting")
        return
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    # Reliability diagram (top)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect calibration')
    ax1.scatter(confidences, accuracies, s=[c/5 for c in counts], 
                alpha=0.7, c='red', label='SolarKnowledge bins')
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title(f'SolarKnowledge Reliability Diagram (ECE = {results["ece"]:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Confidence gap (bottom)
    gaps = [conf - acc for conf, acc in zip(confidences, accuracies)]
    ax2.bar(bin_centers, gaps, width=0.05, alpha=0.7, color='red')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Mean Predicted Probability')
    ax2.set_ylabel('Confidence - Accuracy')
    ax2.set_title('Per-bin Confidence Gap')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = project_root / "solarknowledge_reliability_validation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Reliability plot saved to: {output_path}")
    
    return output_path

def main():
    """Run paper metric validation."""
    print("🧪 PAPER SOLARKNOWLEDGE METRIC VALIDATION")
    print("="*60)
    print("Validating specific metrics claimed in the paper:")
    print("• ECE = 0.185 (from EVEREST comparison)")
    print("• Over-confidence at p ≳ 0.83 (89% confidence, 50% accuracy)")
    print("• Precision struggles (qualitative)")
    
    try:
        # Load test data
        X_test, y_test = load_test_data()
        
        # Load model
        model = load_retrained_solarknowledge()
        
        # Validate metrics
        results = validate_paper_metrics(model, X_test, y_test)
        
        # Print validation results
        print_paper_validation_results(results)
        
        # Create reliability plot
        plot_path = create_reliability_plot(results)
        
        # Save validation summary
        summary_file = project_root / "paper_metric_validation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("PAPER SOLARKNOWLEDGE METRIC VALIDATION SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write("PAPER CLAIMS:\n")
            f.write("• ECE = 0.185\n")
            f.write("• Over-confidence at p ≳ 0.83 (89% confidence, 50% accuracy)\n")
            f.write("• Precision struggles\n\n")
            f.write("MEASURED RESULTS:\n")
            f.write(f"• ECE = {results['ece']:.6f}\n")
            f.write(f"• Precision = {results['precision']:.4f}\n")
            f.write(f"• Accuracy = {results['accuracy']:.4f}\n")
            if results['overconf_083']:
                oc = results['overconf_083']
                f.write(f"• At p > 0.83: {oc['mean_confidence']:.1%} confidence, {oc['actual_accuracy']:.1%} accuracy\n")
            f.write(f"\nSamples: {results['n_samples']}\n")
            f.write(f"Positives: {results['n_positives']}\n")
        
        print(f"\n💾 Validation summary saved to: {summary_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main() 