#!/usr/bin/env python3
"""
BASELINE SOLARKNOWLEDGE TEST EVALUATION
Test an untrained SolarKnowledge model to establish baseline performance characteristics.
This will show what SolarKnowledge's architectural tendency is without any training.
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
    
    # Convert to one-hot for categorical crossentropy
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=2)
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"One-hot labels shape: {y_test_onehot.shape}")
    print(f"Positive class samples: {y_test.sum()}/{len(y_test)} ({100*y_test.sum()/len(y_test):.2f}%)")
    
    return X_test, y_test, y_test_onehot

def create_untrained_solarknowledge():
    """Create an untrained SolarKnowledge model for baseline testing."""
    print("Creating untrained SolarKnowledge model...")
    
    from SolarKnowledge_model import SolarKnowledge
    
    # Initialize model
    model = SolarKnowledge(early_stopping_patience=10)
    
    # Build model with standard parameters
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
        use_focal_loss=False  # Use standard loss for baseline
    )
    
    print("✅ Untrained SolarKnowledge model created successfully!")
    print(f"Model parameters: {model.model.count_params():,}")
    return model

def evaluate_baseline_model(model, X_test, y_test, y_test_onehot):
    """Evaluate untrained model to show baseline behavior."""
    print("Evaluating untrained SolarKnowledge baseline...")
    
    # Get predictions
    y_pred_probs = model.predict(X_test, batch_size=512, verbose=0)
    
    # Extract probabilities for positive class
    y_pred_probs_pos = y_pred_probs[:, 1]
    
    # Convert to binary predictions (threshold = 0.5)
    y_pred = (y_pred_probs_pos > 0.5).astype(int)
    
    # Calculate metrics
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
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'balanced_accuracy': balanced_acc,
        'tss': tss,
        'false_alarm_rate': false_alarm_rate,
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

def analyze_architectural_behavior(results):
    """Analyze what the untrained SolarKnowledge architecture does by default."""
    print("\n" + "="*80)
    print("UNTRAINED SOLARKNOWLEDGE ARCHITECTURAL BEHAVIOR ANALYSIS")
    print("="*80)
    
    print(f"\n📊 BASELINE PERFORMANCE (NO TRAINING):")
    print(f"   Accuracy:         {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"   Precision:        {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"   Recall:           {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"   F1-Score:         {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)")
    print(f"   Balanced Accuracy: {results['balanced_accuracy']:.4f} ({results['balanced_accuracy']*100:.2f}%)")
    print(f"   TSS:              {results['tss']:.4f} ({results['tss']*100:.2f}%)")
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
        
        print(f"\n💡 ARCHITECTURAL BEHAVIOR:")
        total_predictions = tp + fp
        if total_predictions > 0:
            print(f"   • Makes {total_predictions:,} positive predictions ({100*total_predictions/results['n_samples']:.2f}% of data)")
            print(f"   • {fp}/{total_predictions} positive predictions are false alarms ({100*fp/total_predictions:.1f}%)")
        else:
            print(f"   • Makes NO positive predictions (extremely conservative)")
        
        if results['precision'] == 0:
            print(f"   • ZERO precision - all positive predictions are wrong")
        elif results['precision'] < 0.5:
            print(f"   • POOR precision - most positive predictions are wrong")
        else:
            print(f"   • Reasonable precision for untrained model")

def main():
    """Run baseline SolarKnowledge evaluation."""
    print("🧪 BASELINE SOLARKNOWLEDGE EVALUATION")
    print("="*60)
    print("Testing untrained SolarKnowledge to understand architectural behavior")
    
    try:
        # Load test data
        X_test, y_test, y_test_onehot = load_test_data()
        
        # Create untrained model
        model = create_untrained_solarknowledge()
        
        # Evaluate baseline
        results = evaluate_baseline_model(model, X_test, y_test, y_test_onehot)
        
        # Analyze behavior
        analyze_architectural_behavior(results)
        
        # Save results
        results_file = project_root / "solarknowledge_baseline_results.txt"
        with open(results_file, 'w') as f:
            f.write("UNTRAINED SOLARKNOWLEDGE BASELINE PERFORMANCE\n")
            f.write("="*50 + "\n\n")
            f.write("This shows the untrained SolarKnowledge architectural behavior\n")
            f.write("without any training - useful for understanding model biases.\n\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall: {results['recall']:.4f}\n")
            f.write(f"F1-Score: {results['f1_score']:.4f}\n")
            f.write(f"TSS: {results['tss']:.4f}\n")
            f.write(f"False Alarm Rate: {results['false_alarm_rate']:.4f}\n")
            f.write(f"Probability Range: [{results['prob_range'][0]:.3f}, {results['prob_range'][1]:.3f}]\n")
            f.write(f"Mean Probability: {results['prob_mean']:.3f}\n")
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