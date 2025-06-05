#!/usr/bin/env python3
"""
Cost-loss analysis and error analysis for EVEREST M5-72h model.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.solarknowledge_ret_plus import RETPlusWrapper
from models.utils import get_testing_data

def calculate_tss(y_true, y_pred):
    """Calculate True Skill Statistic."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return sensitivity + specificity - 1

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'tss': recall + specificity - 1
    }

def calculate_cost_loss(y_true, y_proba, cost_fn_fp_ratio=20):
    """Calculate expected cost for different thresholds."""
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        cost_fp = 1.0
        cost_fn = cost_fn_fp_ratio * cost_fp
        
        total_cost = fn * cost_fn + fp * cost_fp
        costs.append(total_cost)
    
    costs = np.array(costs)
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    min_cost = costs[optimal_idx]
    
    return thresholds, costs, optimal_threshold, min_cost

def balanced_score(metrics, weights=None):
    """Calculate balanced score using TSS, F1, precision, recall, specificity."""
    if weights is None:
        weights = {'tss': 0.4, 'f1': 0.2, 'precision': 0.15, 'recall': 0.15, 'specificity': 0.1}
    
    score = (weights['tss'] * metrics['tss'] + 
             weights['f1'] * metrics['f1'] + 
             weights['precision'] * metrics['precision'] + 
             weights['recall'] * metrics['recall'] + 
             weights['specificity'] * metrics['specificity'])
    
    return score

def find_optimal_balanced_threshold(y_true, y_proba):
    """Find threshold that maximizes balanced score."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_score = -1
    best_threshold = 0.5
    best_metrics = None
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        metrics = calculate_metrics(y_true, y_pred)
        score = balanced_score(metrics)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics
    
    return best_threshold, best_score, best_metrics

def create_cost_loss_figure(y_true, y_proba, cost_ratio=20):
    """Create cost-loss analysis figure."""
    
    thresholds, costs, optimal_threshold, min_cost = calculate_cost_loss(y_true, y_proba, cost_ratio)
    
    balanced_threshold, balanced_score_val, balanced_metrics = find_optimal_balanced_threshold(y_true, y_proba)
    
    cost_metrics = calculate_metrics(y_true, (y_proba >= optimal_threshold).astype(int))
    balanced_cost_idx = np.where(np.abs(thresholds - balanced_threshold) < 0.01)[0]
    balanced_cost = costs[balanced_cost_idx[0]] if len(balanced_cost_idx) > 0 else costs[50]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(thresholds, costs, 'b-', linewidth=3, alpha=0.8)
    ax1.plot(optimal_threshold, min_cost, 'o', color='orange', markersize=12, 
             markeredgecolor='black', markeredgewidth=2, 
             label=f'Cost-optimal: τ* = {optimal_threshold:.3f}')
    ax1.plot(balanced_threshold, balanced_cost, 's', color='purple', markersize=12,
             markeredgecolor='black', markeredgewidth=2,
             label=f'Thesis-optimal: τ = {balanced_threshold:.3f}')
    
    ax1.set_xlabel('Threshold (τ)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Expected Cost', fontsize=12, fontweight='bold')
    ax1.set_title(f'Cost-Loss Analysis\n(Cost ratio FN:FP = {cost_ratio}:1)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    cost_reduction = ((balanced_cost - min_cost) / balanced_cost) * 100
    
    ax1.text(0.02, 0.98, f'Cost reduction: {cost_reduction:.1f}%\nMin cost: {min_cost:.0f}\nThesis cost: {balanced_cost:.0f}', 
            transform=ax1.transAxes, fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            verticalalignment='top')
    
    comparison_data = {
        'Threshold Strategy': ['Cost-Optimal', 'Thesis-Optimal'],
        'Threshold': [f'{optimal_threshold:.3f}', f'{balanced_threshold:.3f}'],
        'Total Cost': [f'{min_cost:.0f}', f'{balanced_cost:.0f}'],
        'TSS': [f'{cost_metrics["tss"]:.3f}', f'{balanced_metrics["tss"]:.3f}'],
        'F1': [f'{cost_metrics["f1"]:.3f}', f'{balanced_metrics["f1"]:.3f}'],
        'Precision': [f'{cost_metrics["precision"]:.3f}', f'{balanced_metrics["precision"]:.3f}'],
        'Recall': [f'{cost_metrics["recall"]:.3f}', f'{balanced_metrics["recall"]:.3f}']
    }
    
    table_text = 'Threshold Strategy Comparison:\n\n'
    for key in comparison_data:
        if key != 'Threshold Strategy':
            table_text += f'{key}:\n'
            table_text += f'  Cost-optimal: {comparison_data[key][0]}\n'
            table_text += f'  Thesis-optimal: {comparison_data[key][1]}\n\n'
    
    ax2.text(0.05, 0.95, table_text, transform=ax2.transAxes, fontsize=11, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax2.set_title('Performance Comparison', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    
    return fig, {
        'cost_optimal_threshold': optimal_threshold,
        'cost_optimal_cost': min_cost,
        'thesis_optimal_threshold': balanced_threshold,
        'thesis_optimal_cost': balanced_cost,
        'cost_reduction_percent': cost_reduction,
        'cost_metrics': cost_metrics,
        'balanced_metrics': balanced_metrics
    }

def create_confusion_matrices_figure(y_true, y_proba, cost_optimal_threshold, thesis_optimal_threshold):
    """Create confusion matrices comparison figure."""
    
    y_pred_cost = (y_proba >= cost_optimal_threshold).astype(int)
    y_pred_thesis = (y_proba >= thesis_optimal_threshold).astype(int)
    
    cost_metrics = calculate_metrics(y_true, y_pred_cost)
    thesis_metrics = calculate_metrics(y_true, y_pred_thesis)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    cm_thesis = confusion_matrix(y_true, y_pred_thesis)
    sns.heatmap(cm_thesis, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['No Flare', 'Flare'], yticklabels=['No Flare', 'Flare'])
    ax1.set_title(f'Thesis-Optimal Threshold\nτ = {thesis_optimal_threshold:.3f}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    thesis_text = f"""Performance:
TSS: {thesis_metrics['tss']:.3f}
F1: {thesis_metrics['f1']:.3f}
Precision: {thesis_metrics['precision']:.3f}
Recall: {thesis_metrics['recall']:.3f}
Specificity: {thesis_metrics['specificity']:.3f}

FP: {thesis_metrics['FP']}
FN: {thesis_metrics['FN']}"""
    
    ax1.text(1.02, 0.5, thesis_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    cm_cost = confusion_matrix(y_true, y_pred_cost)
    sns.heatmap(cm_cost, annot=True, fmt='d', cmap='Oranges', ax=ax2,
                xticklabels=['No Flare', 'Flare'], yticklabels=['No Flare', 'Flare'])
    ax2.set_title(f'Cost-Optimal Threshold\nτ* = {cost_optimal_threshold:.3f}', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    cost_text = f"""Performance:
TSS: {cost_metrics['tss']:.3f}
F1: {cost_metrics['f1']:.3f}
Precision: {cost_metrics['precision']:.3f}
Recall: {cost_metrics['recall']:.3f}
Specificity: {cost_metrics['specificity']:.3f}

FP: {cost_metrics['FP']}
FN: {cost_metrics['FN']}"""
    
    ax2.text(1.02, 0.5, cost_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    return fig, thesis_metrics, cost_metrics

def main():
    """Main analysis function."""
    
    model_path = '../../tests/model_weights_EVEREST_72h_M5.pt'
    flare_class = 'M5'
    time_window = 72
    
    print("Loading EVEREST model and data...")
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    try:
        test_data = get_testing_data(time_window, flare_class)
        X_test = test_data[0]
        y_test = np.array(test_data[1])
        input_shape = (X_test.shape[1], X_test.shape[2])
        print(f"Data loaded: {X_test.shape}, {len(y_test)} labels, {np.sum(y_test)} positive")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
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
        return
    
    try:
        print("Generating predictions...")
        y_proba = wrapper.predict_proba(X_test)
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.ravel()
        
        print(f"Predictions: [{y_proba.min():.4f}, {y_proba.max():.4f}], mean: {y_proba.mean():.4f}")
        
    except Exception as e:
        print(f"Error generating predictions: {e}")
        return
    
    thesis_threshold = 0.46
    thesis_metrics = calculate_metrics(y_test, (y_proba >= thesis_threshold).astype(int))
    
    print(f"\nThesis threshold (τ = {thesis_threshold}): TSS = {thesis_metrics['tss']:.3f}")
    print(f"Confusion matrix: TP={thesis_metrics['TP']}, TN={thesis_metrics['TN']}, FP={thesis_metrics['FP']}, FN={thesis_metrics['FN']}")
    
    os.makedirs('figs', exist_ok=True)
    
    print("\nCreating cost-loss analysis...")
    cost_fig, cost_results = create_cost_loss_figure(y_test, y_proba, cost_ratio=20)
    
    cost_filename = 'figs/cost_loss.pdf'
    cost_fig.savefig(cost_filename, dpi=300, bbox_inches='tight')
    plt.close(cost_fig)
    print(f"Saved: {cost_filename}")
    
    print(f"\nCost-loss analysis results:")
    print(f"Cost-optimal threshold: {cost_results['cost_optimal_threshold']:.3f}")
    print(f"Thesis-optimal threshold: {cost_results['thesis_optimal_threshold']:.3f}")
    print(f"Cost reduction: {cost_results['cost_reduction_percent']:.1f}%")
    
    print("\nCreating confusion matrices comparison...")
    cm_fig, thesis_detailed, cost_detailed = create_confusion_matrices_figure(
        y_test, y_proba, 
        cost_results['cost_optimal_threshold'], 
        cost_results['thesis_optimal_threshold']
    )
    
    cm_filename = 'figs/confusion_matrices.pdf'
    cm_fig.savefig(cm_filename, dpi=300, bbox_inches='tight')
    plt.close(cm_fig)
    print(f"Saved: {cm_filename}")
    
    summary_text = f"""
Error Analysis and Cost-Loss Results for M5-72h Model

Threshold Comparison:
• Thesis-optimal (balanced scoring): τ = {cost_results['thesis_optimal_threshold']:.3f}
• Cost-optimal (20:1 FN:FP ratio): τ* = {cost_results['cost_optimal_threshold']:.3f}

Performance Comparison:
                    Thesis    Cost-Optimal
TSS                 {thesis_detailed['tss']:.3f}     {cost_detailed['tss']:.3f}
F1-Score           {thesis_detailed['f1']:.3f}     {cost_detailed['f1']:.3f}
Precision          {thesis_detailed['precision']:.3f}     {cost_detailed['precision']:.3f}
Recall             {thesis_detailed['recall']:.3f}     {cost_detailed['recall']:.3f}
Specificity        {thesis_detailed['specificity']:.3f}     {cost_detailed['specificity']:.3f}

Confusion Matrices:
Thesis-optimal: TP={thesis_detailed['TP']}, TN={thesis_detailed['TN']}, FP={thesis_detailed['FP']}, FN={thesis_detailed['FN']}
Cost-optimal:   TP={cost_detailed['TP']}, TN={cost_detailed['TN']}, FP={cost_detailed['FP']}, FN={cost_detailed['FN']}

Cost Analysis:
• Thesis threshold cost: {cost_results['thesis_optimal_cost']:.0f}
• Cost-optimal cost: {cost_results['cost_optimal_cost']:.0f}
• Cost reduction: {cost_results['cost_reduction_percent']:.1f}%

Key Insights:
• Cost-optimal threshold reduces false negatives at expense of more false positives
• {cost_results['cost_reduction_percent']:.1f}% cost reduction available through threshold adjustment
• Trade-off between balanced performance (thesis) and operational cost minimization
"""
    
    latex_section = f"""
\\subsection{{Cost-Loss Analysis}}

Figure~\\ref{{fig:cost_loss}} presents cost-loss analysis for asymmetric misclassification costs with ratio $C_\\text{{FN}}\\!:\\!C_\\text{{FP}}=20\\!:\\!1$, reflecting the operational reality that missing a significant flare (false negative) carries substantially higher consequences than issuing an unnecessary alert (false positive). Under this cost structure, the optimal threshold shifts to $\\tau^\\star={cost_results['cost_optimal_threshold']:.3f}$, compared to the thesis-optimized threshold $\\tau = {cost_results['thesis_optimal_threshold']:.3f}$ derived from balanced scoring.

\\begin{{figure}}[ht]\\centering
\\includegraphics[width=0.75\\linewidth]{{figs/cost_loss.pdf}}
\\caption{{Cost--loss analysis for the M5-72h model under asymmetric cost ratio $C_\\text{{FN}}=20\\,C_\\text{{FP}}$. Orange circle indicates the minimum-cost threshold $\\tau^\\star = {cost_results['cost_optimal_threshold']:.3f}$, while purple square shows the thesis-optimized threshold $\\tau = {cost_results['thesis_optimal_threshold']:.3f}$ from weighted scoring optimization. The cost-optimized threshold prioritizes false negative avoidance at the expense of increased false positive rate.}}
\\label{{fig:cost_loss}}
\\end{{figure}}

\\subsection{{Error Pattern Analysis}}

Figure~\\ref{{fig:confusion_matrices}} contrasts confusion matrices under both threshold strategies. The thesis-optimized threshold ($\\tau = {cost_results['thesis_optimal_threshold']:.3f}$) balances multiple performance dimensions through weighted scoring, while the cost-optimized threshold ($\\tau^\\star = {cost_results['cost_optimal_threshold']:.3f}$) minimizes expected operational cost. This comparison reveals the sensitivity of error patterns to threshold selection and demonstrates the model's adaptability to different operational priorities.

\\begin{{figure}}[ht]\\centering
\\includegraphics[width=0.75\\linewidth]{{figs/confusion_matrices.pdf}}
\\caption{{Confusion matrices comparing thesis-optimized threshold ($\\tau = {cost_results['thesis_optimal_threshold']:.3f}$) with cost-optimized threshold ($\\tau^\\star = {cost_results['cost_optimal_threshold']:.3f}$) for the M5-72h model. The cost-optimized configuration reduces false negatives at the expense of increased false positives, reflecting the 20:1 cost asymmetry. Performance metrics demonstrate the trade-off between balanced accuracy and cost-sensitive optimization.}}
\\label{{fig:confusion_matrices}}
\\end{{figure}}

The error analysis reveals {cost_detailed['FP']} false positives and {cost_detailed['FN']} false negatives under cost-optimal thresholding, compared to {thesis_detailed['FP']} false positives and {thesis_detailed['FN']} false negatives for thesis-optimal thresholding. The threshold comparison demonstrates EVEREST's operational flexibility through probabilistic outputs, enabling adaptation to different stakeholder priorities without retraining.
"""
    
    print(summary_text)
    print("\n" + "="*60)
    print("LATEX SECTION:")
    print("="*60)
    print(latex_section)
    
    with open('figs/error_analysis_summary.txt', 'w') as f:
        f.write(summary_text)
    
    with open('figs/error_analysis_latex.txt', 'w') as f:
        f.write(latex_section)
    
    try:
        import shutil
        shutil.copy('figs/cost_loss.pdf', '../../figs/cost_loss.pdf')
        shutil.copy('figs/confusion_matrices.pdf', '../../figs/confusion_matrices.pdf')
        print("Figures copied to ../../figs/")
    except Exception as e:
        print(f"Could not copy to main figs: {e}")

if __name__ == "__main__":
    main() 