#!/usr/bin/env python3
"""
CORRECTED RETRAINED SOLARKNOWLEDGE TEST EVALUATION
Test the retrained SolarKnowledge M5-24h model on the correct M5-24h test data.
This provides the proper model-to-data match for accurate evaluation.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, balanced_accuracy_score, confusion_matrix,
                           classification_report)
import warnings
warnings.filterwarnings('ignore')

# Add models to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "models"))

def load_test_data():
    """Load SHARP M5-24h test data (correct match for M5-24h model)."""
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

def load_retrained_solarknowledge():
    """Load the retrained SolarKnowledge M5-24h model."""
    print("Loading retrained SolarKnowledge M5-24h model...")
    
    # Path to retrained model
    model_path = project_root / "models/models/SolarKnowledge retrained/Solarknowledge-v4.5_10582_20250531_070617_716154-M5-24h"
    weight_path = model_path / "model_weights.weights.h5"
    
    if not weight_path.exists():
        raise FileNotFoundError(f"Retrained model weights not found at: {weight_path}")
    
    from SolarKnowledge_model import SolarKnowledge
    
    # Initialize model with same parameters as metadata
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
    
    # Compile model with focal loss (as per metadata)
    model.compile(
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        learning_rate=0.0001,
        use_focal_loss=True
    )
    
    try:
        # Load the retrained weights
        model.model.load_weights(str(weight_path))
        print("✅ Retrained SolarKnowledge model loaded successfully!")
        print(f"Model parameters: {model.model.count_params():,}")
        
        # Read metadata for context
        metadata_path = model_path / "metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"Model version: {metadata['version']}")
            print(f"Training accuracy: {metadata['performance']['final_training_accuracy']:.4f}")
            print(f"Epochs trained: {metadata['performance']['epochs_trained']}")
        
        return model
        
    except Exception as e:
        print(f"Failed to load retrained model: {e}")
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

def evaluate_retrained_model(model, X_test, y_test):
    """Evaluate retrained model and return comprehensive metrics."""
    print("Evaluating retrained SolarKnowledge model on M5-24h test data...")
    
    # Get predictions
    y_pred_probs = model.predict(X_test, batch_size=512, verbose=0)
    
    # Handle categorical output
    if len(y_pred_probs.shape) > 1 and y_pred_probs.shape[1] > 1:
        y_pred_probs_pos = y_pred_probs[:, 1]  # Take positive class probability
    else:
        y_pred_probs_pos = y_pred_probs.flatten()
    
    # Convert to binary predictions (threshold = 0.5)
    y_pred = (y_pred_probs_pos > 0.5).astype(int)
    
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
    ece = calculate_ece(y_test, y_pred_probs_pos)
    
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
        'probabilities': y_pred_probs_pos,
        'n_samples': len(y_test),
        'n_positives': y_test.sum(),
        'n_negatives': len(y_test) - y_test.sum(),
        'prob_range': (y_pred_probs_pos.min(), y_pred_probs_pos.max()),
        'prob_mean': y_pred_probs_pos.mean(),
        'prob_std': y_pred_probs_pos.std()
    }

def print_comprehensive_results(results):
    """Print comprehensive evaluation results."""
    print("\n" + "="*80)
    print("RETRAINED SOLARKNOWLEDGE M5-24h TEST PERFORMANCE (CORRECT MATCH)")
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
    """Run corrected retrained SolarKnowledge evaluation."""
    print("🧪 CORRECTED RETRAINED SOLARKNOWLEDGE EVALUATION")
    print("="*60)
    print("Testing retrained SolarKnowledge M5-24h model on correct M5-24h test data")
    print("This provides proper model-to-data matching for accurate evaluation")
    
    try:
        # Load test data
        X_test, y_test = load_test_data()
        
        # Load retrained model
        model = load_retrained_solarknowledge()
        
        # Evaluate
        results = evaluate_retrained_model(model, X_test, y_test)
        
        # Print results
        print_comprehensive_results(results)
        
        # Save results
        results_file = project_root / "retrained_solarknowledge_m5_24h_results.txt"
        with open(results_file, 'w') as f:
            f.write("RETRAINED SOLARKNOWLEDGE M5-24h TEST PERFORMANCE (CORRECT MATCH)\n")
            f.write("="*65 + "\n\n")
            f.write("Model: Retrained SolarKnowledge v4.5 M5-24h\n")
            f.write("Test Data: M5-24h real SHARP data (correct match)\n")
            f.write("Training Pipeline: Official SolarKnowledge training\n\n")
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
        
        # Summary for comparison with EVEREST
        print(f"\n📋 SUMMARY FOR PAPER COMPARISON:")
        print(f"   Retrained SolarKnowledge M5-24h ECE: {results['ece']:.6f}")
        print(f"   EVEREST M5-72h ECE (previous): 0.036306")
        print(f"   Model-Data Match: ✅ Proper M5-24h ↔ M5-24h evaluation")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main() 