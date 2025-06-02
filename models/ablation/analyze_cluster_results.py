"""
Analyze Cluster Ablation Results

This script analyzes ablation results in the format:
models/ablation/results/ablation_[variant]_seed[N]/results.json
"""

import json
import os
import pandas as pd
import numpy as np
import glob


def load_cluster_results(base_dir="models/ablation/results"):
    """Load all cluster ablation results."""

    print("🔬 EVEREST Cluster Ablation Results Analysis")
    print("=" * 60)
    print(f"📂 Loading results from: {base_dir}")

    # Find all results.json files
    pattern = os.path.join(base_dir, "ablation_*", "results.json")
    result_files = glob.glob(pattern)

    # Also check nested structure (in case of double nesting)
    nested_pattern = os.path.join(base_dir, "**", "ablation_*", "results.json")
    nested_files = glob.glob(nested_pattern, recursive=True)

    all_files = list(set(result_files + nested_files))

    print(f"🔍 Found {len(all_files)} result files:")
    for f in all_files:
        print(f"   {f}")

    if not all_files:
        print("❌ No result files found!")
        print("   Expected format: ablation_[variant]_seed[N]/results.json")
        return pd.DataFrame(), {}

    # Load all results
    results = []

    for file_path in all_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Extract experiment info from path
            path_parts = file_path.split(os.sep)
            experiment_dir = None
            for part in path_parts:
                if part.startswith("ablation_"):
                    experiment_dir = part
                    break

            if experiment_dir:
                # Parse experiment name: ablation_[variant]_seed[N]
                parts = experiment_dir.split("_")
                if len(parts) >= 3:
                    variant = "_".join(parts[1:-1])  # Everything between 'ablation' and 'seedN'
                    seed = parts[-1].replace("seed", "")
                else:
                    variant = "unknown"
                    seed = "0"
            else:
                variant = "unknown"
                seed = "0"

            # Create result entry
            result = {
                'file_path': file_path,
                'experiment_name': data.get('experiment_name', experiment_dir),
                'variant_name': variant,
                'seed': int(seed) if seed.isdigit() else 0,
                'best_epoch': data.get('best_epoch', 0),
                'total_epochs': data.get('total_epochs', 0)
            }

            # Add final metrics
            final_metrics = data.get('final_metrics', {})
            for metric, value in final_metrics.items():
                result[f'final_{metric}'] = value

            # Add config info
            config = data.get('config', {})
            variant_config = config.get('variant_config', {})
            for key, value in variant_config.items():
                result[f'config_{key}'] = value

            results.append(result)

        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")

    df = pd.DataFrame(results)

    # Summary by variant
    summary = {}
    if not df.empty:
        print(f"\n📊 Loaded {len(df)} experiments")
        print(f"   Unique variants: {df['variant_name'].nunique()}")
        print(f"   Seeds per variant: {df.groupby('variant_name')['seed'].nunique().to_dict()}")

        # Create summary by variant
        for variant in df['variant_name'].unique():
            variant_df = df[df['variant_name'] == variant]

            if 'final_tss' in variant_df.columns:
                summary[variant] = {
                    'count': len(variant_df),
                    'seeds': sorted(variant_df['seed'].tolist()),
                    'tss_mean': variant_df['final_tss'].mean(),
                    'tss_std': variant_df['final_tss'].std(),
                    'tss_values': variant_df['final_tss'].tolist()
                }

                # Add other metrics if available
                for metric in ['accuracy', 'f1', 'brier', 'ece']:
                    col_name = f'final_{metric}'
                    if col_name in variant_df.columns:
                        summary[variant][f'{metric}_mean'] = variant_df[col_name].mean()
                        summary[variant][f'{metric}_std'] = variant_df[col_name].std()

    return df, summary


def analyze_results(df, summary):
    """Analyze the loaded results."""

    if df.empty:
        print("❌ No data to analyze!")
        return

    print(f"\n📈 RESULTS ANALYSIS")
    print("=" * 50)

    # Check for baseline (full_model)
    baseline_variants = ['full_model', 'full', 'baseline']
    baseline = None

    for variant in baseline_variants:
        if variant in summary:
            baseline = summary[variant]
            baseline_name = variant
            break

    if baseline:
        print(f"🎯 Baseline ({baseline_name}):")
        print(f"   Experiments: {baseline['count']}")
        print(f"   Seeds: {baseline['seeds']}")
        print(f"   TSS: {baseline['tss_mean']:.4f} ± {baseline['tss_std']:.4f}")
        if 'brier_mean' in baseline:
            print(f"   Brier: {baseline['brier_mean']:.4f} ± {baseline['brier_std']:.4f}")
        if 'ece_mean' in baseline:
            print(f"   ECE: {baseline['ece_mean']:.4f} ± {baseline['ece_std']:.4f}")

    print(f"\n📊 All Variants:")
    print("-" * 50)

    # Sort variants by TSS
    sorted_variants = sorted(summary.items(), key=lambda x: x[1]['tss_mean'], reverse=True)

    for variant, stats in sorted_variants:
        print(f"{variant:20} ({stats['count']:2d} exp): TSS {stats['tss_mean']:.4f} ± {stats['tss_std']:.4f}")

        # Compare to baseline if available
        if baseline and variant != baseline_name:
            tss_diff = stats['tss_mean'] - baseline['tss_mean']
            effect = "↑ better" if tss_diff > 0 else "↓ worse" if tss_diff < 0 else "→ same"
            significance = "**" if abs(tss_diff) > 0.05 else "*" if abs(tss_diff) > 0.01 else ""
            print(f"{'':20}             vs baseline: {tss_diff:+.4f} {effect} {significance}")

    # Component effects analysis
    if baseline:
        print(f"\n🔍 COMPONENT EFFECTS (vs {baseline_name}):")
        print("-" * 50)

        component_map = {
            'no_evidential': 'Evidential Head',
            'no_evt': 'EVT Head',
            'no_precursor': 'Precursor Head',
            'mean_pool': 'Attention Bottleneck',
            'cross_entropy': 'Focal Loss (gamma=0)',
            'fp32_training': 'Mixed Precision'
        }

        for variant, stats in summary.items():
            if variant != baseline_name and variant in component_map:
                component = component_map[variant]
                tss_effect = baseline['tss_mean'] - stats['tss_mean']

                # Effect interpretation
                if tss_effect > 0.05:
                    impact = "🔴 Critical"
                elif tss_effect > 0.01:
                    impact = "🟡 Important"
                elif tss_effect > 0.001:
                    impact = "🟢 Minor"
                else:
                    impact = "⚪ Minimal"

                print(f"{component:20}: {tss_effect:+.4f} {impact}")

    return sorted_variants


def create_summary_table(summary):
    """Create a summary table."""

    data = []
    for variant, stats in summary.items():
        row = {
            'Variant': variant,
            'Experiments': stats['count'],
            'Seeds': len(stats['seeds']),
            'TSS_Mean': stats['tss_mean'],
            'TSS_Std': stats['tss_std']
        }

        # Add other metrics if available
        for metric in ['accuracy', 'f1', 'brier', 'ece']:
            if f'{metric}_mean' in stats:
                row[f'{metric.title()}_Mean'] = stats[f'{metric}_mean']
                row[f'{metric.title()}_Std'] = stats[f'{metric}_std']

        data.append(row)

    df = pd.DataFrame(data)
    df = df.sort_values('TSS_Mean', ascending=False)

    print(f"\n📋 SUMMARY TABLE")
    print("=" * 80)
    print(df.round(4).to_string(index=False))

    return df


def main():
    """Run complete analysis."""

    # Try multiple possible locations
    possible_dirs = [
        "models/ablation/results",
        "models/ablation/models/ablation/results",  # In case of nesting
        "ablation/results",
        "results"
    ]

    df = None
    summary = None

    for results_dir in possible_dirs:
        if os.path.exists(results_dir):
            print(f"🔍 Trying directory: {results_dir}")
            df, summary = load_cluster_results(results_dir)
            if not df.empty:
                break

    if df is None or df.empty:
        print("❌ No results found in any expected location!")
        print("Expected directories:", possible_dirs)
        return

    # Analyze results
    sorted_variants = analyze_results(df, summary)

    # Create summary table
    summary_df = create_summary_table(summary)

    # Save analysis
    try:
        output_file = "models/ablation/cluster_results_analysis.json"
        os.makedirs("models/ablation", exist_ok=True)

        analysis_data = {
            'summary_by_variant': summary,
            'sorted_results': {variant: stats for variant, stats in sorted_variants},
            'total_experiments': len(df),
            'unique_variants': len(summary)
        }

        with open(output_file, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)

        print(f"\n💾 Analysis saved to: {output_file}")

    except Exception as e:
        print(f"⚠️ Could not save analysis: {e}")


if __name__ == "__main__":
    main()
