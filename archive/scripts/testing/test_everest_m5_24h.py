#!/usr/bin/env python3
"""
EVEREST M5-24h TEST EVALUATION
Test EVEREST model on M5-24h test data for fair comparison with retrained SolarKnowledge M5-24h.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, balanced_accuracy_score, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# Add models to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "models"))

def load_test_data():
    """Load SHARP M5-24h test data."""
    print("Loading SHARP M5-24h test data...")
    test_data_path = project_root / "Nature_data/testing_data_M5_24.csv"
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

def load_everest_model():
    """Load EVEREST model weights from the correct location."""
    print("Loading EVEREST model...")
    
    # Look for EVEREST model weights
    everest_weights_path = project_root / "tests/model_weights_EVEREST_72h_M5.pt"
    
    if not everest_weights_path.exists():
        raise FileNotFoundError(f"EVEREST model weights not found at: {everest_weights_path}")
    
    from solarknowledge_ret_plus import RETPlusWrapper
    
    # Initialize EVEREST with standard input shape
    model = RETPlusWrapper(input_shape=(10, 9))
    
    # Build the model architecture
    dummy_input = torch.randn(1, 10, 9)
    _ = model.model(dummy_input)  # Build model
    
    # Load weights
    try:
        # Load state dict
        checkpoint = torch.load(everest_weights_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.model.load_state_dict(checkpoint)
        
        model.model.eval()
        print("✅ EVEREST model loaded successfully!")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        return model
        
    except Exception as e:
        print(f"Failed to load EVEREST model: {e}")
        raise

def calculate_ece(y_true, y_pred_probs, n_bins=15):
    """Calculate Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (y_pred_probs > bin_lower) & (y_pred_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Calculate accuracy and confidence for this bin
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred_probs[in_bin].mean()
            
            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def evaluate_everest_model(model, X_test, y_test):
    """Evaluate EVEREST model and return comprehensive metrics."""
    print("Evaluating EVEREST model on M5-24h test data...")
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X_test)
    
    # Get predictions
    with torch.no_grad():
        model.model.eval()
        outputs = model.model(X_tensor)
        
        # Extract probabilities from composite output
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            # Assume first output is main prediction
            y_pred_logits = outputs[0]
        else:
            y_pred_logits = outputs
        
        # Apply sigmoid to get probabilities
        y_pred_probs = torch.sigmoid(y_pred_logits).squeeze().numpy()
    
    # Convert to binary predictions (threshold = 0.5)
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    # Calculate standard metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    # Calculate TSS
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        tss = sensitivity + specificity - 1
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    else:
        tss = 0
        false_alarm_rate = 0
    
    # Calculate ECE
    ece = calculate_ece(y_test, y_pred_probs)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'balanced_accuracy': balanced_acc,
        'tss': tss,
        'false_alarm_rate': false_alarm_rate,
        'ece': ece,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_probs,
        'n_samples': len(y_test),
        'n_positives': y_test.sum(),
        'n_negatives': len(y_test) - y_test.sum(),
        'prob_range': (y_pred_probs.min(), y_pred_probs.max()),
        'prob_mean': y_pred_probs.mean(),
        'prob_std': y_pred_probs.std()
    }

def print_results(results):
    """Print comprehensive evaluation results."""
    print("\n" + "="*80)
    print("EVEREST M5-24h TEST PERFORMANCE")
    print("="*80)
    
    print(f"\n📊 CLASSIFICATION METRICS:")
    print(f"   Accuracy:         {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"   Precision:        {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"   Recall:           {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"   F1-Score:         {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)")
    print(f"   Balanced Accuracy: {results['balanced_accuracy']:.4f} ({results['balanced_accuracy']*100:.2f}%)")
    print(f"   TSS:              {results['tss']:.4f} ({results['tss']*100:.2f}%)")
    
    print(f"\n🎯 CALIBRATION METRICS:")
    print(f"   Expected Calibration Error (ECE): {results['ece']:.6f}")
    
    print(f"\n⚠️  ALARM ANALYSIS:")
    print(f"   False Alarm Rate: {results['false_alarm_rate']:.4f} ({results['false_alarm_rate']*100:.2f}%)")
    
    print(f"\n🔍 PREDICTION ANALYSIS:")
    print(f"   Probability Range: [{results['prob_range'][0]:.3f}, {results['prob_range'][1]:.3f}]")
    print(f"   Mean Probability:  {results['prob_mean']:.3f}")
    print(f"   Std Probability:   {results['prob_std']:.3f}")
    
    print(f"\n📈 DATA DISTRIBUTION:")
    print(f"   Total Samples:    {results['n_samples']:,}")
    print(f"   Positive Samples: {results['n_positives']:,} ({100*results['n_positives']/results['n_samples']:.2f}%)")
    print(f"   Negative Samples: {results['n_negatives']:,} ({100*results['n_negatives']/results['n_samples']:.2f}%)")
    
    print(f"\n🔍 CONFUSION MATRIX:")
    cm = results['confusion_matrix']
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"   True Negatives:  {tn:,}")
        print(f"   False Positives: {fp:,}")
        print(f"   False Negatives: {fn:,}")
        print(f"   True Positives:  {tp:,}")
        
        print(f"\n💡 OPERATIONAL ANALYSIS:")
        total_predictions = tp + fp
        if total_predictions > 0:
            print(f"   • Makes {total_predictions:,} positive predictions ({100*total_predictions/results['n_samples']:.2f}% of data)")
            false_alarm_percentage = (fp / total_predictions) * 100
            print(f"   • {false_alarm_percentage:.1f}% of positive predictions are false alarms")
        else:
            print(f"   • Makes NO positive predictions (extremely conservative)")
            
        if fn > 0 and (tp + fn) > 0:
            missed_percentage = (fn / (tp + fn)) * 100
            print(f"   • {missed_percentage:.1f}% of actual flares are missed")

def main():
    """Run EVEREST M5-24h evaluation."""
    print("🧪 EVEREST M5-24h EVALUATION")
    print("="*60)
    print("Testing EVEREST model on M5-24h test data for fair comparison")
    
    try:
        # Load test data
        X_test, y_test = load_test_data()
        
        # Load EVEREST model
        model = load_everest_model()
        
        # Evaluate
        results = evaluate_everest_model(model, X_test, y_test)
        
        # Print results
        print_results(results)
        
        # Save results
        results_file = project_root / "everest_m5_24h_results.txt"
        with open(results_file, 'w') as f:
            f.write("EVEREST M5-24h TEST PERFORMANCE\n")
            f.write("="*40 + "\n\n")
            f.write("Model: EVEREST (trained on M5-72h, tested on M5-24h)\n")
            f.write("Test Data: M5-24h real SHARP data\n\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall: {results['recall']:.4f}\n")
            f.write(f"F1-Score: {results['f1_score']:.4f}\n")
            f.write(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}\n")
            f.write(f"TSS: {results['tss']:.4f}\n")
            f.write(f"ECE: {results['ece']:.6f}\n")
            f.write(f"False Alarm Rate: {results['false_alarm_rate']:.4f}\n")
            f.write(f"Probability Range: [{results['prob_range'][0]:.3f}, {results['prob_range'][1]:.3f}]\n")
            f.write(f"Total Samples: {results['n_samples']}\n")
            f.write(f"Positive Samples: {results['n_positives']}\n")
            
        print(f"\n💾 Results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main() 