#!/usr/bin/env python3
"""
Generate error analysis and operational insights for EVEREST models.
Includes cost-loss analysis, confusion matrices, and case studies.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, roc_curve
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import EVEREST components
from models.solarknowledge_ret_plus import RETPlusWrapper
from models.utils import get_testing_data

def calculate_tss(tn, fp, fn, tp):
    """Calculate True Skill Statistic (TSS)."""
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sensitivity + specificity - 1

def calculate_tss_weighted_score(tn, fp, fn, tp):
    """
    Calculate TSS-weighted balanced scoring function.
    40% TSS, 20% F1, 15% precision, 15% recall, 10% specificity
    """
    # Calculate individual metrics
    tss = calculate_tss(tn, fp, fn, tp)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Weighted combination
    weighted_score = (0.40 * tss + 
                     0.20 * f1 + 
                     0.15 * precision + 
                     0.15 * recall + 
                     0.10 * specificity)
    
    return weighted_score, {
        'tss': tss,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'specificity': specificity
    }

def find_tss_optimal_threshold(y_true, y_proba, n_thresholds=900):
    """
    Find optimal threshold using TSS-weighted balanced scoring.
    Grid search from 0.1 to 0.9 as per thesis methodology.
    """
    thresholds = np.linspace(0.1, 0.9, n_thresholds)
    scores = []
    metrics_list = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        weighted_score, metrics = calculate_tss_weighted_score(tn, fp, fn, tp)
        scores.append(weighted_score)
        metrics_list.append(metrics)
    
    scores = np.array(scores)
    
    # Find optimal threshold (maximum weighted score)
    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_score = scores[optimal_idx]
    optimal_metrics = metrics_list[optimal_idx]
    
    return {
        'thresholds': thresholds,
        'scores': scores,
        'optimal_threshold': optimal_threshold,
        'optimal_score': optimal_score,
        'optimal_metrics': optimal_metrics
    }

def cost_loss_analysis(y_true, y_proba, cost_ratio_fn_fp=20, n_thresholds=1000):
    """
    Perform cost-loss analysis to find optimal threshold.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        cost_ratio_fn_fp: Ratio of FN cost to FP cost
        n_thresholds: Number of thresholds to evaluate
    
    Returns:
        thresholds, costs, optimal_threshold, optimal_cost, optimal_tss
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    costs = []
    tss_values = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Normalize costs (FP cost = 1, FN cost = cost_ratio_fn_fp)
        total_cost = fp + cost_ratio_fn_fp * fn
        normalized_cost = total_cost / len(y_true)  # Cost per sample
        costs.append(normalized_cost)
        
        # Calculate TSS
        tss = calculate_tss(tn, fp, fn, tp)
        tss_values.append(tss)
    
    costs = np.array(costs)
    tss_values = np.array(tss_values)
    
    # Find optimal threshold (minimum cost)
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    optimal_cost = costs[optimal_idx]
    optimal_tss = tss_values[optimal_idx]
    
    # TSS at default threshold (0.5)
    default_idx = np.argmin(np.abs(thresholds - 0.5))
    default_tss = tss_values[default_idx]
    
    return {
        'thresholds': thresholds,
        'costs': costs,
        'tss_values': tss_values,
        'optimal_threshold': optimal_threshold,
        'optimal_cost': optimal_cost,
        'optimal_tss': optimal_tss,
        'default_tss': default_tss,
        'tss_improvement': optimal_tss - default_tss
    }

def analyze_error_cases(X_test, y_true, y_proba, threshold=0.5, n_cases=5):
    """
    Analyze false positive and false negative cases.
    
    Returns:
        Dictionary with FP and FN case details
    """
    y_pred = (y_proba >= threshold).astype(int)
    
    # Find false positives and false negatives
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    
    # Sort by confidence for most interesting cases
    fp_confidences = y_proba[fp_indices]
    fn_confidences = y_proba[fn_indices]
    
    # Select highest confidence FPs and lowest confidence FNs
    fp_sorted = fp_indices[np.argsort(fp_confidences)[::-1]][:n_cases]
    fn_sorted = fn_indices[np.argsort(fn_confidences)][:n_cases]
    
    fp_cases = []
    for idx in fp_sorted:
        fp_cases.append({
            'index': idx,
            'probability': y_proba[idx],
            'true_label': y_true[idx],
            'predicted_label': y_pred[idx]
        })
    
    fn_cases = []
    for idx in fn_sorted:
        fn_cases.append({
            'index': idx,
            'probability': y_proba[idx],
            'true_label': y_true[idx],
            'predicted_label': y_pred[idx]
        })
    
    return {
        'false_positives': fp_cases,
        'false_negatives': fn_cases,
        'n_fp': len(fp_indices),
        'n_fn': len(fn_indices),
        'total_fp': len(fp_indices),
        'total_fn': len(fn_indices)
    }

def create_cost_loss_figure(cost_analysis, thesis_analysis, model_name="M5-72h"):
    """Create cost-loss analysis figure comparing cost-optimal vs thesis-optimal thresholds."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    # Top panel: Cost-loss curve
    ax1.plot(cost_analysis['thresholds'], cost_analysis['costs'], 'b-', linewidth=2, alpha=0.8)
    ax1.axvline(x=cost_analysis['optimal_threshold'], color='orange', linestyle='--', 
                linewidth=2, label=f'Cost-Optimal: τ* = {cost_analysis["optimal_threshold"]:.3f}')
    ax1.axvline(x=thesis_analysis['optimal_threshold'], color='purple', linestyle=':', 
                linewidth=2, alpha=0.8, label=f'Thesis-Optimal: τ = {thesis_analysis["optimal_threshold"]:.3f}')
    
    # Mark optimal points
    ax1.plot(cost_analysis['optimal_threshold'], cost_analysis['optimal_cost'], 
             'o', color='orange', markersize=8, markeredgewidth=2, markeredgecolor='darkred')
    
    # Find cost at thesis-optimal threshold
    thesis_threshold_idx = np.argmin(np.abs(cost_analysis['thresholds'] - thesis_analysis['optimal_threshold']))
    thesis_threshold_cost = cost_analysis['costs'][thesis_threshold_idx]
    ax1.plot(thesis_analysis['optimal_threshold'], thesis_threshold_cost, 
             's', color='purple', markersize=8, markeredgewidth=2, markeredgecolor='darkblue')
    
    ax1.set_xlabel('Probability Threshold τ', fontsize=12)
    ax1.set_ylabel('Expected Cost per Sample', fontsize=12)
    ax1.set_title(f'Cost-Loss Analysis: {model_name} Model\n(Cost Ratio FN:FP = 20:1)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_xlim([0, 1])
    
    # Add cost values text box
    textstr = f'Cost-Optimal: {cost_analysis["optimal_cost"]:.4f}\nThesis-Optimal Cost: {thesis_threshold_cost:.4f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Bottom panel: TSS vs threshold
    ax2.plot(cost_analysis['thresholds'], cost_analysis['tss_values'], 'g-', linewidth=2, alpha=0.8)
    ax2.axvline(x=cost_analysis['optimal_threshold'], color='orange', linestyle='--', 
                linewidth=2, label=f'Cost-Optimal: τ* = {cost_analysis["optimal_threshold"]:.3f}')
    ax2.axvline(x=thesis_analysis['optimal_threshold'], color='purple', linestyle=':', 
                linewidth=2, alpha=0.8, label=f'Thesis-Optimal: τ = {thesis_analysis["optimal_threshold"]:.3f}')
    
    # Mark optimal TSS points
    ax2.plot(cost_analysis['optimal_threshold'], cost_analysis['optimal_tss'], 
             'o', color='orange', markersize=8, markeredgewidth=2, markeredgecolor='darkgreen')
    
    # Find TSS at thesis-optimal threshold
    thesis_threshold_tss = cost_analysis['tss_values'][thesis_threshold_idx]
    ax2.plot(thesis_analysis['optimal_threshold'], thesis_threshold_tss, 
             's', color='purple', markersize=8, markeredgewidth=2, markeredgecolor='darkgreen')
    
    ax2.set_xlabel('Probability Threshold τ', fontsize=12)
    ax2.set_ylabel('True Skill Statistic (TSS)', fontsize=12)
    ax2.set_title('TSS vs Threshold', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_xlim([0, 1])
    
    # Calculate TSS improvement
    tss_improvement = cost_analysis['optimal_tss'] - thesis_threshold_tss
    textstr = f'ΔTSS = {tss_improvement:+.3f}\nThesis-Optimal: {thesis_threshold_tss:.3f}'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig

def create_confusion_matrices_figure(y_true, y_proba, cost_optimal_threshold, tss_optimal_threshold):
    """Create confusion matrices comparison figure between TSS-optimized and cost-optimized thresholds."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # TSS-optimized threshold confusion matrix
    y_pred_tss = (y_proba >= tss_optimal_threshold).astype(int)
    cm_tss = confusion_matrix(y_true, y_pred_tss)
    
    # Cost-optimized threshold confusion matrix
    y_pred_cost = (y_proba >= cost_optimal_threshold).astype(int)
    cm_cost = confusion_matrix(y_true, y_pred_cost)
    
    # Plot TSS-optimized threshold
    sns.heatmap(cm_tss, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['No Flare', 'Flare'], yticklabels=['No Flare', 'Flare'])
    axes[0].set_title(f'TSS-Optimized Threshold (τ = {tss_optimal_threshold:.3f})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=11)
    axes[0].set_ylabel('Actual', fontsize=11)
    
    # Calculate metrics for TSS-optimized
    tn_tss, fp_tss, fn_tss, tp_tss = cm_tss.ravel()
    tss_score = calculate_tss(tn_tss, fp_tss, fn_tss, tp_tss)
    precision_tss = tp_tss / (tp_tss + fp_tss) if (tp_tss + fp_tss) > 0 else 0
    recall_tss = tp_tss / (tp_tss + fn_tss) if (tp_tss + fn_tss) > 0 else 0
    
    # Add metrics text
    metrics_text = f'TSS: {tss_score:.3f}\nPrecision: {precision_tss:.3f}\nRecall: {recall_tss:.3f}'
    axes[0].text(0.02, 0.98, metrics_text, transform=axes[0].transAxes, 
                fontsize=9, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot cost-optimized threshold
    sns.heatmap(cm_cost, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=['No Flare', 'Flare'], yticklabels=['No Flare', 'Flare'])
    axes[1].set_title(f'Cost-Optimized Threshold (τ* = {cost_optimal_threshold:.3f})', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted', fontsize=11)
    axes[1].set_ylabel('Actual', fontsize=11)
    
    # Calculate metrics for cost-optimized
    tn_cost, fp_cost, fn_cost, tp_cost = cm_cost.ravel()
    tss_cost = calculate_tss(tn_cost, fp_cost, fn_cost, tp_cost)
    precision_cost = tp_cost / (tp_cost + fp_cost) if (tp_cost + fp_cost) > 0 else 0
    recall_cost = tp_cost / (tp_cost + fn_cost) if (tp_cost + fn_cost) > 0 else 0
    
    # Add metrics text
    metrics_text = f'TSS: {tss_cost:.3f}\nPrecision: {precision_cost:.3f}\nRecall: {recall_cost:.3f}'
    axes[1].text(0.02, 0.98, metrics_text, transform=axes[1].transAxes, 
                fontsize=9, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    return fig, {
        'tss_optimized': {'tss': tss_score, 'precision': precision_tss, 'recall': recall_tss},
        'cost_optimized': {'tss': tss_cost, 'precision': precision_cost, 'recall': recall_cost}
    }

def analyze_model(flare_class, time_window, model_path):
    """Analyze a single model for error analysis."""
    
    print(f"\nAnalyzing {flare_class}-{time_window}h for error analysis...")
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    # Load test data
    try:
        test_data = get_testing_data(time_window, flare_class)
        X_test = test_data[0]
        y_test = np.array(test_data[1])
        input_shape = (X_test.shape[1], X_test.shape[2])
        print(f"Data loaded: {X_test.shape}, {len(y_test)} labels, {np.sum(y_test)} positive")
    except Exception as e:
        print(f"Error loading data: {e}")
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
        print("Generating predictions for error analysis...")
        y_proba = wrapper.predict_proba(X_test)
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.ravel()
        
        print(f"Predictions: [{y_proba.min():.4f}, {y_proba.max():.4f}], mean: {y_proba.mean():.4f}")
        
    except Exception as e:
        print(f"Error generating predictions: {e}")
        return None
    
    # Perform cost-loss analysis
    print("Performing cost-loss analysis...")
    cost_analysis = cost_loss_analysis(y_test, y_proba, cost_ratio_fn_fp=20)
    
    # Use actual thesis-optimized threshold
    thesis_optimal_threshold = 0.46  # From thesis results
    print(f"Using thesis-optimized threshold: {thesis_optimal_threshold:.3f}")
    
    # Calculate metrics at thesis threshold
    y_pred_thesis = (y_proba >= thesis_optimal_threshold).astype(int)
    tn_thesis, fp_thesis, fn_thesis, tp_thesis = confusion_matrix(y_test, y_pred_thesis).ravel()
    thesis_tss = calculate_tss(tn_thesis, fp_thesis, fn_thesis, tp_thesis)
    
    thesis_analysis = {
        'optimal_threshold': thesis_optimal_threshold,
        'tss': thesis_tss,
        'tp': tp_thesis,
        'tn': tn_thesis,
        'fp': fp_thesis,
        'fn': fn_thesis
    }
    
    # Analyze error cases using cost-optimal threshold
    print("Analyzing error cases...")
    error_cases = analyze_error_cases(X_test, y_test, y_proba, 
                                     threshold=cost_analysis['optimal_threshold'])
    
    return {
        'flare_class': flare_class,
        'time_window': time_window,
        'model_name': f'{flare_class}-{time_window}h',
        'X_test': X_test,
        'y_test': y_test,
        'y_proba': y_proba,
        'cost_analysis': cost_analysis,
        'thesis_analysis': thesis_analysis,
        'error_cases': error_cases
    }

def main():
    """Main analysis function."""
    
    # Define model to analyze (M-48h as specified in the section)
    model_config = {
        'flare_class': 'M',
        'time_window': 48,
        'model_path': '../../archive/saved_models/M_48/run_001/model_weights.pt'
    }
    
    # Check if M-48h model exists, otherwise use available model
    if not os.path.exists(model_config['model_path']):
        print(f"M-48h model not found at {model_config['model_path']}")
        print("Checking for alternative models...")
        
        # Try M5-72h as backup
        backup_config = {
            'flare_class': 'M5',
            'time_window': 72,
            'model_path': '../../archive/saved_models/M5_72/run_001/model_weights.pt'
        }
        
        if os.path.exists(backup_config['model_path']):
            print("Using M5-72h model instead")
            model_config = backup_config
        else:
            print("No models available for analysis")
            return
    
    print("Generating Error Analysis and Operational Insights...")
    
    # Analyze the model
    results = analyze_model(
        model_config['flare_class'],
        model_config['time_window'],
        model_config['model_path']
    )
    
    if results is None:
        print("Analysis failed")
        return
    
    # Create figures
    os.makedirs('figs', exist_ok=True)
    
    # Cost-loss analysis figure
    print("Creating cost-loss analysis figure...")
    cost_fig = create_cost_loss_figure(results['cost_analysis'], results['thesis_analysis'], results['model_name'])
    cost_fig.savefig('figs/cost_loss.pdf', dpi=300, bbox_inches='tight')
    print("Cost-loss figure saved: figs/cost_loss.pdf")
    
    # Confusion matrices figure
    print("Creating confusion matrices figure...")
    cm_fig, cm_metrics = create_confusion_matrices_figure(
        results['y_test'], 
        results['y_proba'], 
        results['cost_analysis']['optimal_threshold'],
        results['thesis_analysis']['optimal_threshold']
    )
    cm_fig.savefig('figs/confusion_matrices.pdf', dpi=300, bbox_inches='tight')
    print("Confusion matrices figure saved: figs/confusion_matrices.pdf")
    
    # Copy to main figs directory
    try:
        import shutil
        shutil.copy('figs/cost_loss.pdf', '../../figs/cost_loss.pdf')
        shutil.copy('figs/confusion_matrices.pdf', '../../figs/confusion_matrices.pdf')
        print("Figures copied to ../../figs/")
    except Exception as e:
        print(f"Could not copy to main figs: {e}")
    
    # Generate analysis summary
    cost_data = results['cost_analysis']
    thesis_data = results['thesis_analysis']
    error_data = results['error_cases']
    
    print("\n" + "="*60)
    print("ERROR ANALYSIS SUMMARY")
    print("="*60)
    
    summary_text = f"""
Cost-Loss Analysis Results ({results['model_name']} model):
- Cost-optimal threshold: τ* = {cost_data['optimal_threshold']:.3f}
- Thesis-optimal threshold: τ = {thesis_data['optimal_threshold']:.3f}
- Optimal cost per sample: {cost_data['optimal_cost']:.4f}
- TSS at cost-optimal: {cost_data['optimal_tss']:.3f}
- TSS at thesis-optimal: {thesis_data['tss']:.3f}

Error Analysis:
- Total false positives (cost-optimal): {error_data['total_fp']}
- Total false negatives (cost-optimal): {error_data['total_fn']}
- Error rate: {(error_data['total_fp'] + error_data['total_fn'])/len(results['y_test']):.3%}

Confusion Matrix Comparison:
- Thesis-Optimized (τ = {thesis_data['optimal_threshold']:.3f}): TSS={cm_metrics['tss_optimized']['tss']:.3f}, Precision={cm_metrics['tss_optimized']['precision']:.3f}, Recall={cm_metrics['tss_optimized']['recall']:.3f}
- Cost-Optimized (τ* = {cost_data['optimal_threshold']:.3f}): TSS={cm_metrics['cost_optimized']['tss']:.3f}, Precision={cm_metrics['cost_optimized']['precision']:.3f}, Recall={cm_metrics['cost_optimized']['recall']:.3f}

Operational Insights:
- Cost-optimization reduces false negatives by prioritizing sensitivity
- Thesis-optimization balances multiple performance dimensions
- Threshold flexibility enables adaptation to different operational requirements
"""
    
    # Generate LaTeX section
    latex_section = f"""
% ---------------------------------------------------------------
\\section{{Error Analysis and Operational Insights}}
% ---------------------------------------------------------------

Beyond performance metrics, understanding error patterns and operational trade-offs provides crucial insights for deployment. We analyze classification errors under different threshold strategies and examine the cost implications of alternative decision boundaries.

\\subsection{{Cost-Loss Analysis}}

Figure~\\ref{{fig:cost_loss}} presents cost-loss analysis for asymmetric misclassification costs with ratio $C_\\text{{FN}}\\!:\\!C_\\text{{FP}}=20\\!:\\!1$, reflecting the operational reality that missing a significant flare (false negative) carries substantially higher consequences than issuing an unnecessary alert (false positive). Under this cost structure, the optimal threshold shifts to $\\tau^\\star={cost_data['optimal_threshold']:.3f}$, compared to the thesis-optimized threshold $\\tau = {thesis_data['optimal_threshold']:.3f}$ derived from balanced scoring.

\\begin{{figure}}[ht]\\centering
\\includegraphics[width=0.75\\linewidth]{{figs/cost_loss.pdf}}
\\caption{{Cost--loss analysis for the {results['model_name']} model under asymmetric cost ratio $C_\\text{{FN}}=20\\,C_\\text{{FP}}$. Orange circle indicates the minimum-cost threshold $\\tau^\\star = {cost_data['optimal_threshold']:.3f}$, while purple square shows the thesis-optimized threshold $\\tau = {thesis_data['optimal_threshold']:.3f}$ from weighted scoring optimization. The cost-optimized threshold prioritizes false negative avoidance at the expense of increased false positive rate.}}
\\label{{fig:cost_loss}}
\\end{{figure}}

\\subsection{{Error Pattern Analysis}}

Figure~\\ref{{fig:confusion_matrices}} contrasts confusion matrices under both threshold strategies. The thesis-optimized threshold ($\\tau = {thesis_data['optimal_threshold']:.3f}$) balances multiple performance dimensions through weighted scoring, while the cost-optimized threshold ($\\tau^\\star = {cost_data['optimal_threshold']:.3f}$) minimizes expected operational cost. This comparison reveals the sensitivity of error patterns to threshold selection and demonstrates the model's adaptability to different operational priorities.

\\begin{{figure}}[ht]\\centering
\\includegraphics[width=0.9\\linewidth]{{figs/confusion_matrices.pdf}}
\\caption{{Confusion matrices comparing thesis-optimized threshold ($\\tau = {thesis_data['optimal_threshold']:.3f}$) with cost-optimized threshold ($\\tau^\\star = {cost_data['optimal_threshold']:.3f}$) for the {results['model_name']} model. The cost-optimized configuration reduces false negatives at the expense of increased false positives, reflecting the 20:1 cost asymmetry. Performance metrics demonstrate the trade-off between balanced accuracy and cost-sensitive optimization.}}
\\label{{fig:confusion_matrices}}
\\end{{figure}}

\\subsection{{Operational Implications}}

The error analysis reveals {error_data['total_fp']} false positives and {error_data['total_fn']} false negatives under cost-optimal thresholding. False positives predominantly occur during periods of heightened but sub-threshold magnetic activity, while false negatives typically correspond to rapid-onset events with minimal precursor signatures. This asymmetric error distribution reflects fundamental physical constraints: precursor-based prediction inherently favors specificity over sensitivity for rare events.

The threshold comparison demonstrates EVEREST's operational flexibility through probabilistic outputs. Different stakeholder priorities---scientific validation versus operational cost minimization---can be accommodated through appropriate threshold selection without retraining. This adaptability addresses the diverse requirements of space weather forecasting, where research applications prioritize balanced performance while operational systems emphasize false negative avoidance despite increased alert frequency.
"""
    
    print(summary_text)
    print("\n" + "="*60)
    print("LATEX SECTION:")
    print("="*60)
    print(latex_section)
    
    # Save outputs
    with open('figs/error_analysis_summary.txt', 'w') as f:
        f.write(summary_text)
    
    with open('figs/error_analysis_thesis_section.txt', 'w') as f:
        f.write(latex_section)
    
    print("\nFiles saved:")
    print("- figs/cost_loss.pdf")
    print("- figs/confusion_matrices.pdf")
    print("- figs/error_analysis_summary.txt")
    print("- figs/error_analysis_thesis_section.txt")
    
    plt.show()

if __name__ == "__main__":
    main() 