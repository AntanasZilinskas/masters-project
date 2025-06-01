"""
Simple Ablation Analysis Runner

This script runs the complete ablation analysis without requiring any command line arguments.
All paths are hardcoded to the standard locations.
"""

import os
import sys
from pathlib import Path

# Add the current directory to path for imports - works in both scripts and notebooks
try:
    # For scripts
    current_dir = os.path.dirname(__file__)
except NameError:
    # For Jupyter notebooks
    current_dir = os.getcwd()

# Add both the ablation directory and project root to path
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..', '..'))  # Add project root

from models.ablation.analysis_metrics import (
    load_experiment_results,
    compute_baseline_metrics,
    compute_component_ablation_effects,
    compute_overall_improvements,
    compute_calibration_gap_compression,
    compute_brier_99th_analysis,
    generate_summary_report
)

def run_ablation_analysis():
    """Run complete ablation analysis with default settings."""
    
    # Default paths
    results_dir = "models/ablation/results"
    output_dir = "models/ablation/analysis"
    
    print("🔬 EVEREST Ablation Analysis")
    print("=" * 50)
    
    # Load all experiment results
    print(f"📂 Loading results from: {results_dir}")
    df = load_experiment_results(results_dir)
    
    if len(df) == 0:
        print("❌ No experiment results found!")
        print("   Make sure your cluster jobs have completed and saved results to:")
        print(f"   {os.path.abspath(results_dir)}")
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
    report = generate_summary_report(baseline, effects, overall, calibration, brier_99th_effects, output_dir)
    
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
    
    print(f"\n✅ Analysis complete! Results saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    run_ablation_analysis() 