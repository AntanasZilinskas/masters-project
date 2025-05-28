#!/usr/bin/env python3
"""
Demo Publication Results Generator for EVEREST Ablation Study

Shows what the publication results will look like with sample data.
This demonstrates the exact format you'll get once the ablation study is complete.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_demo_data():
    """Generate realistic demo data for ablation study."""
    
    # Set random seed for reproducible demo
    np.random.seed(42)
    
    # Realistic TSS values based on typical solar flare prediction performance
    # Full model should be best, with realistic degradation for ablations
    demo_data = {
        'full_model': {
            'base_tss': 0.2456,
            'std': 0.0123,
            'param_diff': 0
        },
        'no_evidential': {
            'base_tss': 0.2234,  # Slight degradation
            'std': 0.0156,
            'param_diff': -512  # 128 * 4 parameters removed
        },
        'no_evt': {
            'base_tss': 0.2189,  # Moderate degradation
            'std': 0.0134,
            'param_diff': -256  # 128 * 2 parameters removed
        },
        'mean_pool': {
            'base_tss': 0.2087,  # Significant degradation (attention is important)
            'std': 0.0167,
            'param_diff': -64   # 64 * 1 parameters removed
        },
        'cross_entropy': {
            'base_tss': 0.1923,  # Large degradation (focal loss is important)
            'std': 0.0189,
            'param_diff': -768  # Both evidential and EVT heads removed
        },
        'no_precursor': {
            'base_tss': 0.2398,  # Minimal degradation
            'std': 0.0145,
            'param_diff': -128  # 128 * 1 parameters removed
        },
        'fp32_training': {
            'base_tss': 0.2445,  # Very slight degradation (precision effect)
            'std': 0.0134,
            'param_diff': 0     # Same architecture
        }
    }
    
    # Generate 5 seeds per variant
    ablation_models = []
    for variant, config in demo_data.items():
        for seed in range(5):
            # Add realistic noise to TSS values
            tss = config['base_tss'] + np.random.normal(0, config['std'])
            
            ablation_models.append({
                'variant': variant,
                'seed': seed,
                'tss': tss,
                'accuracy': 0.95 + np.random.normal(0, 0.01),  # High accuracy typical for imbalanced data
                'roc_auc': 0.85 + np.random.normal(0, 0.02),
                'brier': 0.05 + np.random.normal(0, 0.005),
                'param_diff': config['param_diff']
            })
    
    return ablation_models

def bootstrap_significance_test(baseline_scores, variant_scores, n_bootstrap=10000):
    """Bootstrap significance test (same as in main script)."""
    if len(baseline_scores) == 0 or len(variant_scores) == 0:
        return np.nan
    
    obs_diff = np.mean(variant_scores) - np.mean(baseline_scores)
    
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        boot_baseline = np.random.choice(baseline_scores, size=len(baseline_scores), replace=True)
        boot_variant = np.random.choice(variant_scores, size=len(variant_scores), replace=True)
        boot_diff = np.mean(boot_variant) - np.mean(boot_baseline)
        bootstrap_diffs.append(boot_diff)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    if obs_diff >= 0:
        p_value = 2 * np.mean(bootstrap_diffs <= -abs(obs_diff))
    else:
        p_value = 2 * np.mean(bootstrap_diffs >= abs(obs_diff))
    
    return min(p_value, 1.0)

def generate_demo_publication_table():
    """Generate demo publication table."""
    
    demo_data = generate_demo_data()
    df = pd.DataFrame(demo_data)
    
    # Calculate statistics for each variant
    variant_stats = []
    
    # Get baseline scores
    baseline_df = df[df['variant'] == 'full_model']
    baseline_tss = baseline_df['tss'].values
    
    display_names = {
        'full_model': 'Full Model (Baseline)',
        'no_evidential': 'No Evidential Head',
        'no_evt': 'No EVT Head', 
        'mean_pool': 'Mean Pooling',
        'cross_entropy': 'Cross-Entropy Loss',
        'no_precursor': 'No Precursor Head',
        'fp32_training': 'FP32 Training'
    }
    
    for variant in ['full_model', 'no_evidential', 'no_evt', 'mean_pool', 'cross_entropy', 'no_precursor', 'fp32_training']:
        variant_df = df[df['variant'] == variant]
        
        if len(variant_df) == 0:
            continue
            
        tss_scores = variant_df['tss'].values
        param_diff = variant_df['param_diff'].iloc[0]
        
        mean_tss = np.mean(tss_scores)
        std_tss = np.std(tss_scores, ddof=1)
        
        # Calculate p-value vs baseline
        if variant == 'full_model':
            p_value = np.nan
        else:
            p_value = bootstrap_significance_test(baseline_tss, tss_scores)
        
        variant_stats.append({
            'variant': variant,
            'display_name': display_names[variant],
            'mean_tss': mean_tss,
            'std_tss': std_tss,
            'param_diff': param_diff,
            'p_value': p_value,
            'n_seeds': len(tss_scores),
            'tss_scores': tss_scores
        })
    
    return variant_stats

def format_demo_latex_table(variant_stats):
    """Format demo results as LaTeX table."""
    
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("  \\centering")
    latex_lines.append("  \\caption{Component ablation study results on the validation set")
    latex_lines.append("           (mean $\\pm$~s.d.\\ over five seeds).}")
    latex_lines.append("  \\label{tab:component_ablation}")
    latex_lines.append("  \\begin{tabular}{lcccc}")
    latex_lines.append("  \\toprule")
    latex_lines.append("  Component & $\\Delta$ Params & TSS $\\uparrow$ & s.d. & $p$ \\\\")
    latex_lines.append("  \\midrule")
    
    for stats in variant_stats:
        variant = stats['display_name']
        param_diff = stats['param_diff']
        mean_tss = stats['mean_tss']
        std_tss = stats['std_tss']
        p_value = stats['p_value']
        
        # Format parameter difference
        if param_diff == 0:
            param_str = "—"
        elif param_diff > 0:
            param_str = f"+{param_diff//1000:.0f}k" if param_diff >= 1000 else f"+{param_diff}"
        else:
            param_str = f"{param_diff//1000:.0f}k" if abs(param_diff) >= 1000 else f"{param_diff}"
        
        # Format TSS
        tss_str = f"{mean_tss:.4f}"
        
        # Format standard deviation
        std_str = f"{std_tss:.4f}"
        
        # Format p-value
        if np.isnan(p_value):
            p_str = "—"
        elif p_value < 0.001:
            p_str = "$<$0.001"
        elif p_value < 0.01:
            p_str = f"{p_value:.3f}"
        else:
            p_str = f"{p_value:.2f}"
        
        # Bold the baseline
        if stats['variant'] == 'full_model':
            variant = f"\\textbf{{{variant}}}"
        
        latex_lines.append(f"  {variant:<25} & {param_str:>8} & {tss_str:>6} & {std_str:>6} & {p_str:>8} \\\\")
    
    latex_lines.append("  \\bottomrule")
    latex_lines.append("  \\end{tabular}")
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)

def generate_demo_summary(variant_stats):
    """Generate demo summary statistics."""
    
    summary_lines = []
    summary_lines.append("📊 DEMO: PUBLICATION ABLATION STUDY SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append("(This shows what your results will look like)")
    summary_lines.append("")
    
    # Overall statistics
    total_experiments = sum(stats['n_seeds'] for stats in variant_stats)
    n_variants = len(variant_stats)
    
    summary_lines.append(f"Total experiments: {total_experiments}")
    summary_lines.append(f"Number of variants: {n_variants}")
    summary_lines.append(f"Seeds per variant: 5")
    summary_lines.append("")
    
    # Performance by variant
    summary_lines.append("📈 Performance by Variant:")
    for stats in variant_stats:
        variant = stats['display_name']
        mean_tss = stats['mean_tss']
        std_tss = stats['std_tss']
        p_value = stats['p_value']
        
        p_str = "baseline" if np.isnan(p_value) else f"p={p_value:.3f}"
        summary_lines.append(f"   • {variant:<25}: TSS={mean_tss:.4f}±{std_tss:.4f} ({p_str})")
    
    summary_lines.append("")
    
    # Statistical significance summary
    significant_variants = [s for s in variant_stats if not np.isnan(s['p_value']) and s['p_value'] < 0.05]
    summary_lines.append(f"🎯 Statistically Significant Differences (p < 0.05): {len(significant_variants)}")
    
    baseline_tss = next((s['mean_tss'] for s in variant_stats if s['variant'] == 'full_model'), np.nan)
    
    for stats in significant_variants:
        variant = stats['display_name']
        p_value = stats['p_value']
        mean_tss = stats['mean_tss']
        
        if not np.isnan(baseline_tss):
            diff = mean_tss - baseline_tss
            direction = "↑" if diff > 0 else "↓"
            summary_lines.append(f"   • {variant}: {direction} {abs(diff):.4f} TSS (p={p_value:.3f})")
    
    # Best and worst performers
    summary_lines.append("")
    best_variant = max(variant_stats, key=lambda x: x['mean_tss'])
    worst_variant = min(variant_stats, key=lambda x: x['mean_tss'])
    
    summary_lines.append(f"🏆 Best Performance: {best_variant['display_name']} (TSS={best_variant['mean_tss']:.4f})")
    summary_lines.append(f"📉 Worst Performance: {worst_variant['display_name']} (TSS={worst_variant['mean_tss']:.4f})")
    
    return "\n".join(summary_lines)

def main():
    """Main demo function."""
    
    print("🔬 DEMO: EVEREST Ablation Study - Publication Results")
    print("=" * 80)
    print("This shows exactly what your publication results will look like!")
    print("")
    
    # Generate demo data
    print("📊 Generating demo ablation statistics...")
    variant_stats = generate_demo_publication_table()
    
    # Generate LaTeX table
    print("📝 Formatting LaTeX table...")
    latex_table = format_demo_latex_table(variant_stats)
    
    # Generate summary
    print("📈 Generating summary statistics...")
    summary_stats = generate_demo_summary(variant_stats)
    
    # Save demo results
    results_dir = Path("demo_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save LaTeX table
    latex_file = results_dir / "demo_component_ablation_table.tex"
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    
    # Save summary
    summary_file = results_dir / "demo_ablation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(summary_stats)
    
    # Display results
    print("\n" + summary_stats)
    print("\n📋 LaTeX Table (ready for your paper):")
    print("-" * 50)
    print(latex_table)
    
    print(f"\n✅ Demo results saved to 'demo_results/' directory")
    print(f"📁 Files created:")
    print(f"   • demo_component_ablation_table.tex")
    print(f"   • demo_ablation_summary.txt")
    
    print(f"\n💡 Once you run the actual ablation study, use:")
    print(f"   python generate_publication_results.py")
    print(f"   (This will generate the same format with your real data)")

if __name__ == "__main__":
    main() 