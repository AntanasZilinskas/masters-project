#!/usr/bin/env python3
"""
ACTUAL SOLARKNOWLEDGE TEST EVALUATION
Properly test SolarKnowledge model on real test data to get accurate performance numbers.
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
    """Load the actual SolarKnowledge model."""
    print("Loading SolarKnowledge model...")
    
    # Try different weight paths
    weight_paths = [
        project_root / "models/weights/72/M5/model_weights.weights.h5",
        project_root / "models/archive/models_working/72/M5/model_weights.weights.h5"
    ]
    
    from SolarKnowledge_model import SolarKnowledge
    
    # Initialize model correctly (no input_shape parameter)
    model = SolarKnowledge(early_stopping_patience=10)
    
    # Build model with input shape
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
    
    # Try to load weights
    for weight_path in weight_paths:
        if weight_path.exists():
            print(f"Loading weights from: {weight_path}")
            try:
                model.model.load_weights(str(weight_path))
                print("✅ SolarKnowledge model loaded successfully!")
                print(f"Model parameters: {model.model.count_params():,}")
                return model
            except Exception as e:
                print(f"Failed to load from {weight_path}: {e}")
                continue
    
    raise FileNotFoundError("Could not load SolarKnowledge model weights")

def calculate_tss(y_true, y_pred):
    """Calculate True Skill Statistic (TSS)."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        tss = sensitivity + specificity - 1
        return tss
    return 0

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return comprehensive metrics."""
    print("Evaluating model on test data...")
    
    # Get predictions
    y_pred_probs = model.predict(X_test)
    
    # Handle different output formats
    if len(y_pred_probs.shape) > 1 and y_pred_probs.shape[1] > 1:
        y_pred_probs = y_pred_probs[:, 1]  # Take positive class probability
    else:
        y_pred_probs = y_pred_probs.flatten()
    
    # Convert to binary predictions (threshold = 0.5)
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    tss = calculate_tss(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # False alarm rate
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_positive_rate = 1 - precision if precision > 0 else 1
    else:
        false_alarm_rate = 0
        false_positive_rate = 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'balanced_accuracy': balanced_acc,
        'tss': tss,
        'false_alarm_rate': false_alarm_rate,
        'false_positive_rate': false_positive_rate,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_probs,
        'n_samples': len(y_test),
        'n_positives': y_test.sum(),
        'n_negatives': len(y_test) - y_test.sum()
    }

def print_results(results):
    """Print comprehensive evaluation results."""
    print("\n" + "="*80)
    print("ACTUAL SOLARKNOWLEDGE M5-72h TEST PERFORMANCE")
    print("="*80)
    
    print(f"\n📊 CLASSIFICATION METRICS:")
    print(f"   Accuracy:         {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"   Precision:        {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"   Recall:           {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"   F1-Score:         {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)")
    print(f"   Balanced Accuracy: {results['balanced_accuracy']:.4f} ({results['balanced_accuracy']*100:.2f}%)")
    print(f"   TSS:              {results['tss']:.4f} ({results['tss']*100:.2f}%)")
    
    print(f"\n⚠️  ALARM ANALYSIS:")
    print(f"   False Positive Rate: {results['false_positive_rate']:.4f} ({results['false_positive_rate']*100:.2f}%)")
    print(f"   False Alarm Rate:    {results['false_alarm_rate']:.4f} ({results['false_alarm_rate']*100:.2f}%)")
    
    print(f"\n📈 DATA DISTRIBUTION:")
    print(f"   Total Samples:    {results['n_samples']:,}")
    print(f"   Positive Samples: {results['n_positives']:,} ({100*results['n_positives']/results['n_samples']:.2f}%)")
    print(f"   Negative Samples: {results['n_negatives']:,} ({100*results['n_negatives']/results['n_samples']:.2f}%)")
    
    print(f"\n🔍 CONFUSION MATRIX:")
    cm = results['confusion_matrix']
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"   True Negatives:  {tn:,}")
        print(f"   False Positives: {fp:,} (False Alarms)")
        print(f"   False Negatives: {fn:,} (Missed Flares)")
        print(f"   True Positives:  {tp:,}")
        
        print(f"\n💡 INTERPRETATION:")
        if fp > 0:
            false_alarm_percentage = (fp / (tp + fp)) * 100
            print(f"   • {false_alarm_percentage:.1f}% of positive predictions are false alarms")
        if fn > 0:
            missed_percentage = (fn / (tp + fn)) * 100
            print(f"   • {missed_percentage:.1f}% of actual flares are missed")

def main():
    """Run actual SolarKnowledge test evaluation."""
    print("🧪 ACTUAL SOLARKNOWLEDGE TEST EVALUATION")
    print("="*60)
    print("Testing SolarKnowledge model on real SHARP M5-72h test data")
    
    try:
        # Load test data
        X_test, y_test = load_test_data()
        
        # Load model
        model = load_solarknowledge_model()
        
        # Evaluate
        results = evaluate_model(model, X_test, y_test)
        
        # Print results
        print_results(results)
        
        # Save results
        results_file = project_root / "solarknowledge_actual_test_results.txt"
        with open(results_file, 'w') as f:
            f.write("ACTUAL SOLARKNOWLEDGE M5-72h TEST PERFORMANCE\n")
            f.write("="*50 + "\n\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall: {results['recall']:.4f}\n")
            f.write(f"F1-Score: {results['f1_score']:.4f}\n")
            f.write(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}\n")
            f.write(f"TSS: {results['tss']:.4f}\n")
            f.write(f"False Positive Rate: {results['false_positive_rate']:.4f}\n")
            f.write(f"False Alarm Rate: {results['false_alarm_rate']:.4f}\n")
            f.write(f"\nTest Samples: {results['n_samples']}\n")
            f.write(f"Positive Samples: {results['n_positives']}\n")
            f.write(f"Negative Samples: {results['n_negatives']}\n")
            
        print(f"\n💾 Results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main() 