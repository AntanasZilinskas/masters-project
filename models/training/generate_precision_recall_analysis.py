#!/usr/bin/env python3
"""
Generate comprehensive Precision-Recall analysis for EVEREST.
Better suited for rare event prediction than ROC curves.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, f1_score
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import EVEREST components
from models.solarknowledge_ret_plus import RETPlusWrapper
from models.utils import get_testing_data

def calculate_tss(y_true, y_pred):
    """Calculate True Skill Statistic."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        if (tp + fn) == 0 or (tn + fp) == 0:
            return 0.0
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return sensitivity + specificity - 1
    return 0.0

def find_optimal_thresholds(y_true, y_proba, precision, recall, thresholds):
    """Find optimal thresholds based on different criteria."""
    
    # Calculate F1 scores for each threshold
    f1_scores = []
    tss_scores = []
    
    for i, threshold in enumerate(thresholds):
        y_pred = (y_proba >= threshold).astype(int)
        
        # F1 score
        if len(np.unique(y_pred)) > 1:  # Check if both classes are predicted
            f1 = f1_score(y_true, y_pred)
        else:
            f1 = 0.0
        f1_scores.append(f1)
        
        # TSS score
        tss = calculate_tss(y_true, y_pred)
        tss_scores.append(tss)
    
    f1_scores = np.array(f1_scores)
    tss_scores = np.array(tss_scores)
    
    # Find optimal points
    optimal_f1_idx = np.argmax(f1_scores)
    optimal_tss_idx = np.argmax(tss_scores)
    
    # Find balanced precision-recall point (closest to diagonal)
    pr_distances = np.abs(precision - recall)
    balanced_pr_idx = np.argmin(pr_distances)
    
    # Find high precision operating point (precision >= 0.8)
    high_precision_mask = precision >= 0.8
    if np.any(high_precision_mask):
        high_precision_idx = np.where(high_precision_mask)[0][-1]  # Last index with high precision
    else:
        high_precision_idx = 0
    
    return {
        'f1_optimal': {
            'idx': optimal_f1_idx,
            'threshold': thresholds[optimal_f1_idx],
            'precision': precision[optimal_f1_idx],
            'recall': recall[optimal_f1_idx],
            'f1': f1_scores[optimal_f1_idx],
            'tss': tss_scores[optimal_f1_idx]
        },
        'tss_optimal': {
            'idx': optimal_tss_idx,
            'threshold': thresholds[optimal_tss_idx],
            'precision': precision[optimal_tss_idx],
            'recall': recall[optimal_tss_idx],
            'f1': f1_scores[optimal_tss_idx],
            'tss': tss_scores[optimal_tss_idx]
        },
        'balanced_pr': {
            'idx': balanced_pr_idx,
            'threshold': thresholds[balanced_pr_idx],
            'precision': precision[balanced_pr_idx],
            'recall': recall[balanced_pr_idx],
            'f1': f1_scores[balanced_pr_idx],
            'tss': tss_scores[balanced_pr_idx]
        },
        'high_precision': {
            'idx': high_precision_idx,
            'threshold': thresholds[high_precision_idx],
            'precision': precision[high_precision_idx],
            'recall': recall[high_precision_idx],
            'f1': f1_scores[high_precision_idx],
            'tss': tss_scores[high_precision_idx]
        }
    }

def analyze_model_performance(flare_class, time_window, model_path):
    """Analyze a single model's performance."""
    
    print(f"\n{'='*50}")
    print(f"Analyzing EVEREST {flare_class}-{time_window}h")
    print(f"{'='*50}")
    
    # Load test data
    try:
        test_data = get_testing_data(time_window, flare_class)
        X_test = test_data[0]
        y_test = test_data[1]
        print(f"Test data shape: {X_test.shape}, Labels: {len(y_test)}")
        input_shape = (X_test.shape[1], X_test.shape[2])
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None
    
    # Load model
    try:
        wrapper = RETPlusWrapper(
            input_shape=input_shape,
            early_stopping_patience=10,
            use_attention_bottleneck=True,
            use_evidential=True,
            use_evt=True,
            use_precursor=True,
            compile_model=False
        )
        wrapper.load(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Generate predictions
    try:
        y_proba = wrapper.predict_proba(X_test)
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.ravel()
        
        print(f"Prediction range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")
        print(f"Mean probability: {y_proba.mean():.4f}")
    except Exception as e:
        print(f"Error generating predictions: {e}")
        return None
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    ap_score = average_precision_score(y_test, y_proba)
    
    # Calculate baseline performance
    positive_rate = np.mean(y_test)
    print(f"Positive class rate: {positive_rate:.4f} ({np.sum(y_test)}/{len(y_test)})")
    
    # Find optimal operating points
    optimal_points = find_optimal_thresholds(y_test, y_proba, precision, recall, thresholds)
    
    return {
        'flare_class': flare_class,
        'time_window': time_window,
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'ap_score': ap_score,
        'positive_rate': positive_rate,
        'optimal_points': optimal_points,
        'y_test': y_test,
        'y_proba': y_proba
    }

def create_precision_recall_figure(results_list):
    """Create comprehensive precision-recall analysis figure."""
    
    # Set up the figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Color palette for different models
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # ---- Panel 1: Main Precision-Recall Curves ----
    ax1 = fig.add_subplot(gs[0, :2])
    
    for i, result in enumerate(results_list):
        if result is None:
            continue
            
        precision = result['precision']
        recall = result['recall']
        ap_score = result['ap_score']
        positive_rate = result['positive_rate']
        flare_class = result['flare_class']
        time_window = result['time_window']
        
        # Plot PR curve
        ax1.plot(recall, precision, color=colors[i], linewidth=2.5,
                label=f'{flare_class}-{time_window}h (AP={ap_score:.3f})')
        
        # Plot optimal F1 point
        f1_opt = result['optimal_points']['f1_optimal']
        ax1.plot(f1_opt['recall'], f1_opt['precision'], 'o', 
                color=colors[i], markersize=8, markeredgecolor='white', markeredgewidth=1)
    
    # Add baseline (random classifier)
    if results_list and results_list[0] is not None:
        baseline_ap = results_list[0]['positive_rate']
        ax1.axhline(y=baseline_ap, color='gray', linestyle='--', alpha=0.7,
                   label=f'Random (AP={baseline_ap:.3f})')
    
    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Curves: EVEREST Performance', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    
    # ---- Panel 2: F1-Score vs Threshold ----
    ax2 = fig.add_subplot(gs[0, 2])
    
    for i, result in enumerate(results_list):
        if result is None:
            continue
            
        # Calculate F1 scores for different thresholds
        f1_scores = []
        thresholds_sampled = np.linspace(0.01, 0.99, 50)
        
        for threshold in thresholds_sampled:
            y_pred = (result['y_proba'] >= threshold).astype(int)
            if len(np.unique(y_pred)) > 1:
                f1 = f1_score(result['y_test'], y_pred)
            else:
                f1 = 0.0
            f1_scores.append(f1)
        
        ax2.plot(thresholds_sampled, f1_scores, color=colors[i], linewidth=2,
                label=f'{result["flare_class"]}-{result["time_window"]}h')
        
        # Mark optimal F1 point
        f1_opt = result['optimal_points']['f1_optimal']
        ax2.plot(f1_opt['threshold'], f1_opt['f1'], 'o', 
                color=colors[i], markersize=6)
    
    ax2.set_xlabel('Threshold', fontsize=11)
    ax2.set_ylabel('F1-Score', fontsize=11)
    ax2.set_title('F1-Score vs Threshold', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    
    # ---- Panel 3: Operating Points Analysis ----
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    # Create performance summary table
    if results_list and results_list[0] is not None:
        result = results_list[0]  # Focus on the main result
        
        # Create detailed performance summary
        summary_text = f"""
        EVEREST {result['flare_class']}-{result['time_window']}h Precision-Recall Analysis
        ════════════════════════════════════════════════════════════════════
        
        Overall Performance:
        • Average Precision (AP): {result['ap_score']:.3f}
        • Baseline (Random): {result['positive_rate']:.3f}
        • Improvement over Random: {(result['ap_score']/result['positive_rate'] - 1)*100:.1f}%
        
        Optimal Operating Points:
        ┌────────────────┬───────────┬───────────┬─────────┬─────────┬─────────┐
        │ Criterion      │ Threshold │ Precision │ Recall  │ F1-Score│ TSS     │
        ├────────────────┼───────────┼───────────┼─────────┼─────────┼─────────┤
        │ Max F1-Score   │   {result['optimal_points']['f1_optimal']['threshold']:.3f}   │   {result['optimal_points']['f1_optimal']['precision']:.3f}   │  {result['optimal_points']['f1_optimal']['recall']:.3f}  │  {result['optimal_points']['f1_optimal']['f1']:.3f}  │  {result['optimal_points']['f1_optimal']['tss']:.3f}  │
        │ Max TSS        │   {result['optimal_points']['tss_optimal']['threshold']:.3f}   │   {result['optimal_points']['tss_optimal']['precision']:.3f}   │  {result['optimal_points']['tss_optimal']['recall']:.3f}  │  {result['optimal_points']['tss_optimal']['f1']:.3f}  │  {result['optimal_points']['tss_optimal']['tss']:.3f}  │
        │ Balanced P-R   │   {result['optimal_points']['balanced_pr']['threshold']:.3f}   │   {result['optimal_points']['balanced_pr']['precision']:.3f}   │  {result['optimal_points']['balanced_pr']['recall']:.3f}  │  {result['optimal_points']['balanced_pr']['f1']:.3f}  │  {result['optimal_points']['balanced_pr']['tss']:.3f}  │
        │ High Precision │   {result['optimal_points']['high_precision']['threshold']:.3f}   │   {result['optimal_points']['high_precision']['precision']:.3f}   │  {result['optimal_points']['high_precision']['recall']:.3f}  │  {result['optimal_points']['high_precision']['f1']:.3f}  │  {result['optimal_points']['high_precision']['tss']:.3f}  │
        └────────────────┴───────────┴───────────┴─────────┴─────────┴─────────┘
        
        Operational Recommendations:
        • For Maximum Detection (Emergency Preparedness): Use Max TSS threshold ({result['optimal_points']['tss_optimal']['threshold']:.3f})
        • For Balanced Operations: Use Max F1 threshold ({result['optimal_points']['f1_optimal']['threshold']:.3f})
        • For Minimal False Alarms: Use High Precision threshold ({result['optimal_points']['high_precision']['threshold']:.3f})
        """
        
        ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle('EVEREST Solar Flare Prediction: Precision-Recall Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    return fig

def main():
    """Main analysis function."""
    
    # Define models to analyze
    models_to_analyze = [
        {
            'flare_class': 'M5',
            'time_window': 72,
            'model_path': '../../archive/saved_models/M5_72/run_001/model_weights.pt'
        }
        # Can add more models here for comparison
        # {
        #     'flare_class': 'M',
        #     'time_window': 24,
        #     'model_path': '../../archive/saved_models/M_24/run_001/model_weights.pt'
        # }
    ]
    
    # Analyze each model
    results = []
    for model_config in models_to_analyze:
        if os.path.exists(model_config['model_path']):
            result = analyze_model_performance(
                model_config['flare_class'],
                model_config['time_window'],
                model_config['model_path']
            )
            results.append(result)
        else:
            print(f"Model not found: {model_config['model_path']}")
            results.append(None)
    
    # Create comprehensive figure
    if any(r is not None for r in results):
        fig = create_precision_recall_figure(results)
        
        # Save figure
        os.makedirs('figs', exist_ok=True)
        fig.savefig('figs/precision_recall_analysis.pdf', dpi=300, bbox_inches='tight')
        print(f"\nFigure saved: figs/precision_recall_analysis.pdf")
        
        # Also create a simpler version for main text
        fig2, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        main_result = next(r for r in results if r is not None)
        precision = main_result['precision']
        recall = main_result['recall']
        ap_score = main_result['ap_score']
        positive_rate = main_result['positive_rate']
        
        # Plot main PR curve
        ax.plot(recall, precision, color='#2E86AB', linewidth=3,
               label=f'EVEREST (AP = {ap_score:.3f})')
        
        # Plot optimal F1 point
        f1_opt = main_result['optimal_points']['f1_optimal']
        ax.plot(f1_opt['recall'], f1_opt['precision'], 'o', 
               color='#F24236', markersize=10, markeredgecolor='white', 
               markeredgewidth=2, label=f'Optimal F1 = {f1_opt["f1"]:.3f}')
        
        # Add baseline
        ax.axhline(y=positive_rate, color='gray', linestyle='--', alpha=0.7,
                  label=f'Random (AP = {positive_rate:.3f})')
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall: EVEREST {main_result["flare_class"]}-{main_result["time_window"]}h', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        plt.savefig('figs/precision_recall_simple.pdf', dpi=300, bbox_inches='tight')
        print(f"Simple figure saved: figs/precision_recall_simple.pdf")
        
        plt.show()
    else:
        print("No valid results to plot.")

if __name__ == "__main__":
    main() 