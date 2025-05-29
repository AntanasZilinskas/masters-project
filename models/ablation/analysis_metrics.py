"""
Comprehensive Ablation Analysis for EVEREST

This script computes all requested ablation metrics:
1. Change in TSS from Attention Bottleneck
2. Change in ECE from Evidential NIG Head  
3. Change in 99th-percentile Brier Score from EVT-GPD Tail Head
4. Change in TSS from Precursor Auxiliary Head
5. Overall Mean TSS Improvement
6. Overall Calibration Gap Compression

Usage:
    python analysis_metrics.py --results-dir models/ablation/results
    python analysis_metrics.py --comprehensive  # For all 9 tasks
"""

import os
import json
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from scipy import stats

def load_experiment_results(results_dir: str) -> pd.DataFrame:
    """Load all experiment results into a comprehensive DataFrame."""
    
    results = []
    results_files = glob.glob(os.path.join(results_dir, "**/results.json"), recursive=True)
    
    for file_path in results_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract key information
            result = {
                'experiment_name': data.get('experiment_name'),
                'variant_name': data.get('variant_name'),
                'seed': data.get('seed'),
                'sequence_variant': data.get('sequence_variant'),
                'best_epoch': data.get('best_epoch'),
                'total_epochs': data.get('total_epochs'),
                'file_path': file_path
            }
            
            # Add final metrics
            final_metrics = data.get('final_metrics', {})
            for metric, value in final_metrics.items():
                result[f'final_{metric}'] = value
            
            # Add configuration info
            config = data.get('config', {})
            variant_config = config.get('variant_config', {})
            for key, value in variant_config.items():
                result[f'config_{key}'] = value
            
            results.append(result)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    return pd.DataFrame(results)

def compute_baseline_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute baseline metrics from full model (all components enabled)."""
    
    # Find baseline (full model) experiments
    baseline_mask = (
        (df['config_use_attention_bottleneck'] == True) &
        (df['config_use_evidential'] == True) &
        (df['config_use_evt'] == True) &
        (df['config_use_precursor'] == True) &
        (df['sequence_variant'].isna())  # No sequence variant = standard 10 timesteps
    )
    
    baseline_df = df[baseline_mask]
    
    if len(baseline_df) == 0:
        print("⚠️ No baseline (full model) experiments found!")
        return {}
    
    print(f"📊 Found {len(baseline_df)} baseline experiments")
    
    # Check for 99th percentile Brier score availability
    has_brier_99th = 'final_brier_99th' in baseline_df.columns and not baseline_df['final_brier_99th'].isna().all()
    
    # Compute means across seeds
    baseline_metrics = {
        'tss': baseline_df['final_tss'].mean(),
        'tss_std': baseline_df['final_tss'].std(),
        'ece': baseline_df['final_ece'].mean(),
        'ece_std': baseline_df['final_ece'].std(),
        'brier': baseline_df['final_brier'].mean(),
        'brier_std': baseline_df['final_brier'].std(),
        'brier_99th': baseline_df['final_brier_99th'].mean() if has_brier_99th else None,
        'brier_99th_std': baseline_df['final_brier_99th'].std() if has_brier_99th else None,
        'accuracy': baseline_df['final_accuracy'].mean(),
        'f1': baseline_df['final_f1'].mean(),
        'count': len(baseline_df)
    }
    
    print(f"🎯 Baseline TSS: {baseline_metrics['tss']:.4f} ± {baseline_metrics['tss_std']:.4f}")
    print(f"🎯 Baseline ECE: {baseline_metrics['ece']:.4f} ± {baseline_metrics['ece_std']:.4f}")
    if baseline_metrics['brier_99th'] is not None:
        print(f"🎯 Baseline 99th-percentile Brier: {baseline_metrics['brier_99th']:.4f} ± {baseline_metrics['brier_99th_std']:.4f}")
    else:
        print("⚠️ 99th-percentile Brier score not available in baseline experiments")
        print("   This metric was added in recent trainer updates - re-run experiments to get this metric")
    
    return baseline_metrics

def compute_component_ablation_effects(df: pd.DataFrame, baseline: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Compute the effect of each component ablation."""
    
    effects = {}
    
    # 1. Attention Bottleneck Effect
    no_attention_mask = (
        (df['config_use_attention_bottleneck'] == False) &
        (df['config_use_evidential'] == True) &
        (df['config_use_evt'] == True) &
        (df['config_use_precursor'] == True) &
        (df['sequence_variant'].isna())
    )
    
    no_attention_df = df[no_attention_mask]
    if len(no_attention_df) > 0:
        tss_change = baseline['tss'] - no_attention_df['final_tss'].mean()
        effects['attention_bottleneck'] = {
            'tss_change': tss_change,
            'tss_change_pvalue': stats.ttest_1samp(no_attention_df['final_tss'], baseline['tss']).pvalue,
            'mean_without': no_attention_df['final_tss'].mean(),
            'std_without': no_attention_df['final_tss'].std(),
            'n_experiments': len(no_attention_df)
        }
        print(f"🔄 Attention Bottleneck TSS change: {tss_change:+.4f} (p={effects['attention_bottleneck']['tss_change_pvalue']:.4f})")
    
    # 2. Evidential Head Effect
    no_evidential_mask = (
        (df['config_use_attention_bottleneck'] == True) &
        (df['config_use_evidential'] == False) &
        (df['config_use_evt'] == True) &
        (df['config_use_precursor'] == True) &
        (df['sequence_variant'].isna())
    )
    
    no_evidential_df = df[no_evidential_mask]
    if len(no_evidential_df) > 0:
        ece_change = baseline['ece'] - no_evidential_df['final_ece'].mean()
        effects['evidential_head'] = {
            'ece_change': ece_change,
            'ece_change_pvalue': stats.ttest_1samp(no_evidential_df['final_ece'], baseline['ece']).pvalue,
            'mean_without': no_evidential_df['final_ece'].mean(),
            'std_without': no_evidential_df['final_ece'].std(),
            'n_experiments': len(no_evidential_df)
        }
        print(f"🎲 Evidential Head ECE change: {ece_change:+.4f} (p={effects['evidential_head']['ece_change_pvalue']:.4f})")
    
    # 3. EVT Head Effect (99th percentile Brier score)
    no_evt_mask = (
        (df['config_use_attention_bottleneck'] == True) &
        (df['config_use_evidential'] == True) &
        (df['config_use_evt'] == False) &
        (df['config_use_precursor'] == True) &
        (df['sequence_variant'].isna())
    )
    
    no_evt_df = df[no_evt_mask]
    print(f"🔍 Found {len(no_evt_df)} experiments with EVT head disabled")
    
    # Check if we have the required data for 99th percentile Brier analysis
    has_brier_99th_data = (
        len(no_evt_df) > 0 and 
        'final_brier_99th' in df.columns and 
        not no_evt_df['final_brier_99th'].isna().all() and
        baseline.get('brier_99th') is not None
    )
    
    if has_brier_99th_data:
        brier_99th_change = baseline['brier_99th'] - no_evt_df['final_brier_99th'].mean()
        effects['evt_head'] = {
            'brier_99th_change': brier_99th_change,
            'brier_99th_change_pvalue': stats.ttest_1samp(no_evt_df['final_brier_99th'], baseline['brier_99th']).pvalue,
            'mean_without': no_evt_df['final_brier_99th'].mean(),
            'std_without': no_evt_df['final_brier_99th'].std(),
            'baseline_mean': baseline['brier_99th'],
            'baseline_std': baseline['brier_99th_std'],
            'n_experiments': len(no_evt_df)
        }
        print(f"📊 EVT Head 99th-percentile Brier change: {brier_99th_change:+.4f} (p={effects['evt_head']['brier_99th_change_pvalue']:.4f})")
        print(f"   Baseline: {baseline['brier_99th']:.4f}, Without EVT: {no_evt_df['final_brier_99th'].mean():.4f}")
        
        # Interpretation
        if brier_99th_change > 0:
            print("   ✅ EVT head helps with tail risk (removing it increases 99th-percentile Brier score)")
        else:
            print("   ❌ EVT head may hurt tail risk (removing it decreases 99th-percentile Brier score)")
            
    elif len(no_evt_df) == 0:
        print("⚠️ No experiments found with EVT head disabled - cannot compute EVT effect")
        print("   Make sure you have 'no_evt' variant in your ablation study")
    elif baseline.get('brier_99th') is None:
        print("⚠️ 99th-percentile Brier score not available in baseline")
        print("   This metric requires updated trainer - re-run baseline experiments")
    else:
        print("⚠️ 99th-percentile Brier score not available in EVT ablation experiments")
        print("   Re-run 'no_evt' experiments with updated trainer to get this metric")
    
    # 4. Precursor Head Effect
    no_precursor_mask = (
        (df['config_use_attention_bottleneck'] == True) &
        (df['config_use_evidential'] == True) &
        (df['config_use_evt'] == True) &
        (df['config_use_precursor'] == False) &
        (df['sequence_variant'].isna())
    )
    
    no_precursor_df = df[no_precursor_mask]
    if len(no_precursor_df) > 0:
        tss_change = baseline['tss'] - no_precursor_df['final_tss'].mean()
        effects['precursor_head'] = {
            'tss_change': tss_change,
            'tss_change_pvalue': stats.ttest_1samp(no_precursor_df['final_tss'], baseline['tss']).pvalue,
            'mean_without': no_precursor_df['final_tss'].mean(),
            'std_without': no_precursor_df['final_tss'].std(),
            'n_experiments': len(no_precursor_df)
        }
        print(f"🎯 Precursor Head TSS change: {tss_change:+.4f} (p={effects['precursor_head']['tss_change_pvalue']:.4f})")
    
    return effects

def compute_overall_improvements(df: pd.DataFrame, baseline: Dict[str, float]) -> Dict[str, float]:
    """Compute overall mean improvements across all ablation variants."""
    
    # Get all ablation experiments (excluding baseline)
    ablation_mask = ~(
        (df['config_use_attention_bottleneck'] == True) &
        (df['config_use_evidential'] == True) &
        (df['config_use_evt'] == True) &
        (df['config_use_precursor'] == True) &
        (df['sequence_variant'].isna())
    )
    
    ablation_df = df[ablation_mask]
    
    if len(ablation_df) == 0:
        return {}
    
    # Group by variant and compute means
    variant_means = ablation_df.groupby('variant_name')['final_tss'].mean()
    
    # Compute how much each variant differs from baseline
    tss_improvements = baseline['tss'] - variant_means
    
    overall_stats = {
        'mean_tss_improvement': tss_improvements.mean(),
        'std_tss_improvement': tss_improvements.std(),
        'median_tss_improvement': tss_improvements.median(),
        'max_tss_degradation': tss_improvements.min(),  # Most negative = worst degradation
        'min_tss_degradation': tss_improvements.max(),  # Most positive = least degradation
        'n_variants': len(variant_means)
    }
    
    print(f"\n📈 Overall Performance Summary:")
    print(f"   Mean TSS change: {overall_stats['mean_tss_improvement']:+.4f} ± {overall_stats['std_tss_improvement']:.4f}")
    print(f"   Median TSS change: {overall_stats['median_tss_improvement']:+.4f}")
    print(f"   Worst degradation: {overall_stats['max_tss_degradation']:+.4f}")
    print(f"   Best preservation: {overall_stats['min_tss_degradation']:+.4f}")
    
    return overall_stats

def compute_calibration_gap_compression(df: pd.DataFrame, baseline: Dict[str, float]) -> Dict[str, float]:
    """Compute calibration gap compression across experiments."""
    
    # Get ECE values for all variants
    all_ece = df['final_ece'].dropna()
    
    if len(all_ece) == 0:
        return {}
    
    # Compute range compression
    baseline_ece = baseline['ece']
    
    calibration_stats = {
        'baseline_ece': baseline_ece,
        'min_ece': all_ece.min(),
        'max_ece': all_ece.max(),
        'mean_ece': all_ece.mean(),
        'std_ece': all_ece.std(),
        'ece_range': all_ece.max() - all_ece.min(),
        'range_compression_vs_baseline': (all_ece.max() - all_ece.min()) / (baseline_ece + 1e-8),
        'n_experiments': len(all_ece)
    }
    
    # How much does each component affect calibration?
    component_effects = {}
    
    # Check if evidential head improves calibration
    no_evidential_mask = df['config_use_evidential'] == False
    if no_evidential_mask.sum() > 0:
        no_evidential_ece = df[no_evidential_mask]['final_ece'].mean()
        component_effects['evidential_removal_ece_change'] = no_evidential_ece - baseline_ece
    
    calibration_stats.update(component_effects)
    
    print(f"\n🎯 Calibration Analysis:")
    print(f"   Baseline ECE: {baseline_ece:.4f}")
    print(f"   ECE range: {calibration_stats['ece_range']:.4f}")
    print(f"   Mean ECE: {calibration_stats['mean_ece']:.4f} ± {calibration_stats['std_ece']:.4f}")
    
    return calibration_stats

def compute_brier_99th_analysis(df: pd.DataFrame, baseline: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Compute 99th-percentile Brier score changes for all components."""
    
    print(f"\n📊 99th-Percentile Brier Score Analysis (Tail Risk)")
    print("=" * 50)
    
    if baseline.get('brier_99th') is None:
        print("⚠️ 99th-percentile Brier score not available - skipping tail risk analysis")
        return {}
    
    brier_effects = {}
    
    # Check each component's effect on tail risk
    components = [
        ("attention_bottleneck", "config_use_attention_bottleneck", False),
        ("evidential_head", "config_use_evidential", False), 
        ("evt_head", "config_use_evt", False),
        ("precursor_head", "config_use_precursor", False)
    ]
    
    for comp_name, config_col, target_value in components:
        # Find experiments where this component is disabled
        comp_mask = (
            (df[config_col] == target_value) &
            (df['sequence_variant'].isna())  # No sequence variants
        )
        
        # For the target component, ensure other components are enabled
        if comp_name == "attention_bottleneck":
            comp_mask = comp_mask & (df['config_use_evidential'] == True) & (df['config_use_evt'] == True) & (df['config_use_precursor'] == True)
        elif comp_name == "evidential_head":
            comp_mask = comp_mask & (df['config_use_attention_bottleneck'] == True) & (df['config_use_evt'] == True) & (df['config_use_precursor'] == True)
        elif comp_name == "evt_head":
            comp_mask = comp_mask & (df['config_use_attention_bottleneck'] == True) & (df['config_use_evidential'] == True) & (df['config_use_precursor'] == True)
        elif comp_name == "precursor_head":
            comp_mask = comp_mask & (df['config_use_attention_bottleneck'] == True) & (df['config_use_evidential'] == True) & (df['config_use_evt'] == True)
        
        comp_df = df[comp_mask]
        
        if len(comp_df) > 0 and 'final_brier_99th' in comp_df.columns and not comp_df['final_brier_99th'].isna().all():
            brier_99th_without = comp_df['final_brier_99th'].mean()
            brier_99th_change = baseline['brier_99th'] - brier_99th_without
            
            try:
                pvalue = stats.ttest_1samp(comp_df['final_brier_99th'], baseline['brier_99th']).pvalue
            except:
                pvalue = 1.0
            
            brier_effects[comp_name] = {
                'brier_99th_change': brier_99th_change,
                'brier_99th_pvalue': pvalue,
                'baseline_brier_99th': baseline['brier_99th'],
                'without_component_brier_99th': brier_99th_without,
                'n_experiments': len(comp_df)
            }
            
            # Interpretation
            effect_direction = "↓ reduces" if brier_99th_change > 0 else "↑ increases"
            significance = "*" if pvalue < 0.05 else ""
            
            print(f"{comp_name.replace('_', ' ').title()}: {brier_99th_change:+.4f}{significance} ({effect_direction} tail risk)")
            print(f"   Baseline: {baseline['brier_99th']:.4f} → Without: {brier_99th_without:.4f} (p={pvalue:.4f})")
            
        else:
            print(f"{comp_name.replace('_', ' ').title()}: No data available")
    
    return brier_effects

def generate_summary_report(baseline: Dict, effects: Dict, overall: Dict, calibration: Dict, 
                          brier_99th_effects: Dict, output_dir: str = "models/ablation/analysis"):
    """Generate comprehensive summary report."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        "analysis_timestamp": pd.Timestamp.now().isoformat(),
        "baseline_metrics": baseline,
        "component_effects": effects,
        "overall_improvements": overall,
        "calibration_analysis": calibration,
        "brier_99th_analysis": brier_99th_effects,
        "summary": {
            "attention_bottleneck_tss_change": effects.get('attention_bottleneck', {}).get('tss_change'),
            "evidential_head_ece_change": effects.get('evidential_head', {}).get('ece_change'),
            "evt_head_brier99_change": effects.get('evt_head', {}).get('brier_99th_change'),
            "precursor_head_tss_change": effects.get('precursor_head', {}).get('tss_change'),
            "overall_mean_tss_improvement": overall.get('mean_tss_improvement'),
            "calibration_gap_compression": calibration.get('range_compression_vs_baseline'),
            # Add all component 99th-percentile Brier changes
            "attention_bottleneck_brier99_change": brier_99th_effects.get('attention_bottleneck', {}).get('brier_99th_change'),
            "evidential_head_brier99_change": brier_99th_effects.get('evidential_head', {}).get('brier_99th_change'),
            "evt_head_brier99_change_detailed": brier_99th_effects.get('evt_head', {}).get('brier_99th_change'),
            "precursor_head_brier99_change": brier_99th_effects.get('precursor_head', {}).get('brier_99th_change')
        }
    }
    
    # Save JSON report
    report_path = os.path.join(output_dir, "ablation_analysis_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Generate LaTeX-friendly summary table
    latex_summary = generate_latex_table(report)
    latex_path = os.path.join(output_dir, "ablation_summary_table.tex")
    with open(latex_path, 'w') as f:
        f.write(latex_summary)
    
    # Generate 99th-percentile Brier table
    if brier_99th_effects:
        latex_brier = generate_brier_99th_table(brier_99th_effects, baseline)
        brier_latex_path = os.path.join(output_dir, "brier_99th_analysis_table.tex")
        with open(brier_latex_path, 'w') as f:
            f.write(latex_brier)
        print(f"   LaTeX Brier 99th: {brier_latex_path}")
    
    print(f"\n📋 Analysis reports saved:")
    print(f"   JSON: {report_path}")
    print(f"   LaTeX: {latex_path}")
    
    return report

def generate_brier_99th_table(brier_effects: Dict, baseline: Dict) -> str:
    """Generate LaTeX table for 99th-percentile Brier score analysis."""
    
    latex = """\\begin{table}[htbp]
\\centering
\\caption{99th-Percentile Brier Score Changes by Component (Tail Risk Analysis)}
\\label{tab:brier_99th_analysis}
\\begin{tabular}{lccc}
\\hline
Component & Baseline & Without Component & Change \\\\
\\hline
"""
    
    component_names = {
        'attention_bottleneck': 'Attention Bottleneck',
        'evidential_head': 'Evidential Head',
        'evt_head': 'EVT-GPD Head',
        'precursor_head': 'Precursor Head'
    }
    
    # Add rows for each component
    for comp_key, comp_name in component_names.items():
        if comp_key in brier_effects:
            effect = brier_effects[comp_key]
            baseline_val = effect['baseline_brier_99th']
            without_val = effect['without_component_brier_99th']
            change = effect['brier_99th_change']
            pvalue = effect['brier_99th_pvalue']
            significance = "*" if pvalue < 0.05 else ""
            
            latex += f"{comp_name} & {baseline_val:.4f} & {without_val:.4f} & {change:+.4f}{significance} \\\\\n"
        else:
            latex += f"{comp_name} & \\multicolumn{{3}}{{c}}{{No data available}} \\\\\n"
    
    latex += f"""\\hline
\\end{{tabular}}
\\begin{{tablenotes}}
\\small
\\item Note: * indicates statistical significance (p < 0.05).
\\item Baseline 99th-percentile Brier: {baseline.get('brier_99th', 'N/A'):.4f}
\\item Positive changes indicate component helps with tail risk.
\\item Higher Brier scores = worse calibration for extreme predictions.
\\end{{tablenotes}}
\\end{{table}}"""
    
    return latex

def generate_latex_table(report: Dict) -> str:
    """Generate LaTeX table for the paper."""
    
    effects = report["component_effects"]
    
    latex = """\\begin{table}[htbp]
\\centering
\\caption{Ablation Study Results for EVEREST Components}
\\label{tab:ablation_results}
\\begin{tabular}{lcc}
\\hline
Component & Metric & Change \\\\
\\hline
"""
    
    # Add rows for each component
    if 'attention_bottleneck' in effects:
        change = effects['attention_bottleneck']['tss_change']
        pvalue = effects['attention_bottleneck']['tss_change_pvalue']
        significance = "*" if pvalue < 0.05 else ""
        latex += f"Attention Bottleneck & TSS & {change:+.4f}{significance} \\\\\n"
    
    if 'evidential_head' in effects:
        change = effects['evidential_head']['ece_change']
        pvalue = effects['evidential_head']['ece_change_pvalue']
        significance = "*" if pvalue < 0.05 else ""
        latex += f"Evidential Head & ECE & {change:+.4f}{significance} \\\\\n"
    
    if 'evt_head' in effects:
        change = effects['evt_head']['brier_99th_change']
        pvalue = effects['evt_head']['brier_99th_change_pvalue']
        significance = "*" if pvalue < 0.05 else ""
        latex += f"EVT-GPD Head & 99th-percentile Brier & {change:+.4f}{significance} \\\\\n"
    
    if 'precursor_head' in effects:
        change = effects['precursor_head']['tss_change']
        pvalue = effects['precursor_head']['tss_change_pvalue']
        significance = "*" if pvalue < 0.05 else ""
        latex += f"Precursor Head & TSS & {change:+.4f}{significance} \\\\\n"
    
    latex += """\\hline
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Note: * indicates statistical significance (p < 0.05). 
\\item Positive changes indicate improvement when component is removed.
\\end{tablenotes}
\\end{table}"""
    
    return latex

def main():
    parser = argparse.ArgumentParser(description="Analyze EVEREST ablation study results")
    parser.add_argument("--results-dir", default="models/ablation/results", 
                       help="Directory containing experiment results")
    parser.add_argument("--output-dir", default="models/ablation/analysis",
                       help="Directory to save analysis reports")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Analyze across all 9 flare class/time window combinations")
    
    args = parser.parse_args()
    
    print("🔬 EVEREST Ablation Analysis")
    print("=" * 50)
    
    # Load all experiment results
    print(f"📂 Loading results from: {args.results_dir}")
    df = load_experiment_results(args.results_dir)
    
    if len(df) == 0:
        print("❌ No experiment results found!")
        return
    
    print(f"📊 Loaded {len(df)} experiments")
    print(f"   Variants: {df['variant_name'].nunique()}")
    print(f"   Seeds per variant: {df['seed'].nunique()}")
    print(f"   Unique experiments: {df['experiment_name'].nunique()}")
    
    # Compute baseline metrics
    baseline = compute_baseline_metrics(df)
    
    if not baseline:
        print("❌ Cannot proceed without baseline metrics!")
        return
    
    # Compute component ablation effects
    print(f"\n🔍 Computing component ablation effects...")
    effects = compute_component_ablation_effects(df, baseline)
    
    # Compute overall improvements
    print(f"\n📈 Computing overall improvements...")
    overall = compute_overall_improvements(df, baseline)
    
    # Compute calibration analysis
    print(f"\n🎯 Computing calibration analysis...")
    calibration = compute_calibration_gap_compression(df, baseline)
    
    # Compute 99th-percentile Brier score analysis
    print(f"\n🔍 Computing 99th-percentile Brier score analysis...")
    brier_99th_effects = compute_brier_99th_analysis(df, baseline)
    
    # Generate comprehensive report
    report = generate_summary_report(baseline, effects, overall, calibration, brier_99th_effects, args.output_dir)
    
    # Print final summary
    print(f"\n🎯 FINAL ABLATION RESULTS:")
    print("=" * 50)
    
    summary = report["summary"]
    if summary["attention_bottleneck_tss_change"]:
        print(f"1. Attention Bottleneck TSS change: {summary['attention_bottleneck_tss_change']:+.4f}")
    
    if summary["evidential_head_ece_change"]:
        print(f"2. Evidential Head ECE change: {summary['evidential_head_ece_change']:+.4f}")
    
    if summary["evt_head_brier99_change"]:
        print(f"3. EVT Head 99th-percentile Brier change: {summary['evt_head_brier99_change']:+.4f}")
    
    if summary["precursor_head_tss_change"]:
        print(f"4. Precursor Head TSS change: {summary['precursor_head_tss_change']:+.4f}")
    
    if summary["overall_mean_tss_improvement"]:
        print(f"5. Overall Mean TSS Improvement: {summary['overall_mean_tss_improvement']:+.4f}")
    
    if summary["calibration_gap_compression"]:
        print(f"6. Calibration Gap Compression: {summary['calibration_gap_compression']:.4f}")
    
    # Add comprehensive 99th-percentile Brier analysis
    print(f"\n📊 COMPREHENSIVE 99th-PERCENTILE BRIER ANALYSIS (M5-72h):")
    print("-" * 60)
    brier_components = [
        ("attention_bottleneck_brier99_change", "Attention Bottleneck"),
        ("evidential_head_brier99_change", "Evidential Head"),
        ("evt_head_brier99_change_detailed", "EVT-GPD Head"),
        ("precursor_head_brier99_change", "Precursor Head")
    ]
    
    has_brier_data = False
    for key, name in brier_components:
        if summary.get(key) is not None:
            change = summary[key]
            effect = "↓ reduces tail risk" if change > 0 else "↑ increases tail risk"
            print(f"{name}: {change:+.4f} ({effect})")
            has_brier_data = True
    
    if not has_brier_data:
        print("⚠️ No 99th-percentile Brier data available")
        print("   Run experiments with updated trainer to get tail risk metrics")
    else:
        print("\nNote: Positive changes mean removing the component worsens tail risk")
        print("      (i.e., the component helps with extreme prediction calibration)")

if __name__ == "__main__":
    main() 