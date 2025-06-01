"""
Jupyter Notebook Ablation Analysis

Copy and paste this into a Jupyter notebook cell to run the ablation analysis.
Designed specifically for notebook environments.
"""

# Notebook-friendly imports
import os
import sys
import pandas as pd
import numpy as np
import json
import glob
from pathlib import Path
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

def compute_baseline_metrics(df: pd.DataFrame) -> dict:
    """Compute baseline metrics from full model."""
    
    # Find baseline (full model) experiments
    baseline_mask = (
        (df['config_use_attention_bottleneck'] == True) &
        (df['config_use_evidential'] == True) &
        (df['config_use_evt'] == True) &
        (df['config_use_precursor'] == True) &
        (df['sequence_variant'].isna())
    )
    
    baseline_df = df[baseline_mask]
    
    if len(baseline_df) == 0:
        print("⚠️ No baseline (full model) experiments found!")
        return {}
    
    print(f"📊 Found {len(baseline_df)} baseline experiments")
    
    # Compute means across seeds
    baseline_metrics = {
        'tss': baseline_df['final_tss'].mean(),
        'tss_std': baseline_df['final_tss'].std(),
        'ece': baseline_df['final_ece'].mean(),
        'ece_std': baseline_df['final_ece'].std(),
        'brier': baseline_df['final_brier'].mean(),
        'brier_std': baseline_df['final_brier'].std(),
        'accuracy': baseline_df['final_accuracy'].mean(),
        'f1': baseline_df['final_f1'].mean(),
        'count': len(baseline_df)
    }
    
    # Add 99th percentile Brier if available
    if 'final_brier_99th' in baseline_df.columns and not baseline_df['final_brier_99th'].isna().all():
        baseline_metrics['brier_99th'] = baseline_df['final_brier_99th'].mean()
        baseline_metrics['brier_99th_std'] = baseline_df['final_brier_99th'].std()
    
    print(f"🎯 Baseline TSS: {baseline_metrics['tss']:.4f} ± {baseline_metrics['tss_std']:.4f}")
    print(f"🎯 Baseline ECE: {baseline_metrics['ece']:.4f} ± {baseline_metrics['ece_std']:.4f}")
    
    return baseline_metrics

def run_quick_analysis():
    """Run a quick ablation analysis for Jupyter notebooks."""
    
    # Paths
    results_dir = "models/ablation/results"
    
    print("🔬 EVEREST Ablation Analysis (Jupyter Version)")
    print("=" * 50)
    
    # Load results
    print(f"📂 Loading results from: {results_dir}")
    df = load_experiment_results(results_dir)
    
    if len(df) == 0:
        print("❌ No experiment results found!")
        print("   Make sure your cluster jobs have completed and saved results to:")
        print(f"   {os.path.abspath(results_dir)}")
        return None
    
    print(f"📊 Loaded {len(df)} experiments")
    print(f"   Variants: {df['variant_name'].nunique()}")
    print(f"   Seeds per variant: {df['seed'].nunique()}")
    
    # Show available variants
    print(f"\n🔍 Available variants:")
    for variant in df['variant_name'].unique():
        count = len(df[df['variant_name'] == variant])
        print(f"   {variant}: {count} experiments")
    
    # Compute baseline
    baseline = compute_baseline_metrics(df)
    
    if not baseline:
        print("❌ Cannot proceed without baseline metrics!")
        return df
    
    # Quick component analysis
    print(f"\n📈 Quick Component Analysis:")
    print("-" * 40)
    
    # Analyze each component
    components = [
        ('no_evidential', 'Evidential Head', 'ece'),
        ('no_evt', 'EVT Head', 'brier'),
        ('mean_pool', 'Attention Bottleneck', 'tss'),
        ('no_precursor', 'Precursor Head', 'tss')
    ]
    
    for variant, name, metric in components:
        variant_df = df[df['variant_name'] == variant]
        if len(variant_df) > 0:
            if f'final_{metric}' in variant_df.columns:
                variant_mean = variant_df[f'final_{metric}'].mean()
                baseline_mean = baseline[metric]
                change = baseline_mean - variant_mean
                print(f"{name:20} {metric.upper():>4}: {change:+.4f} (baseline: {baseline_mean:.4f}, without: {variant_mean:.4f})")
            else:
                print(f"{name:20} {metric.upper():>4}: No data available")
        else:
            print(f"{name:20} {metric.upper():>4}: No experiments found")
    
    # Overall TSS summary
    print(f"\n🎯 TSS Summary Across All Variants:")
    print("-" * 40)
    for variant in df['variant_name'].unique():
        variant_df = df[df['variant_name'] == variant]
        if len(variant_df) > 0 and 'final_tss' in variant_df.columns:
            tss_mean = variant_df['final_tss'].mean()
            tss_std = variant_df['final_tss'].std()
            print(f"{variant:20}: {tss_mean:.4f} ± {tss_std:.4f}")
    
    return df

# Run the analysis
if __name__ == "__main__":
    df = run_quick_analysis()
    
    # Display the dataframe if successful
    if df is not None and len(df) > 0:
        print(f"\n📋 Results DataFrame shape: {df.shape}")
        print("Available columns:", df.columns.tolist()) 