#!/usr/bin/env python3
"""
Enhanced Ablation Results Finder with Metadata Support

This version can properly identify ablation models using the enhanced metadata
that includes variant and seed information.
"""

import os
import json
import glob
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd


def find_recent_everest_models(hours_back=24):
    """Find EVEREST model directories created in the last N hours."""
    cutoff_time = datetime.now() - timedelta(hours=hours_back)

    # Look in both models/models/ and models/ directories (prioritize models/models/)
    search_patterns = [
        "models/models/EVEREST-v*",
        "models/EVEREST-v*",
        "EVEREST-v*"
    ]

    recent_models = []

    for pattern in search_patterns:
        for model_dir in glob.glob(pattern):
            if os.path.isdir(model_dir):
                # Check creation time
                stat = os.stat(model_dir)
                creation_time = datetime.fromtimestamp(stat.st_mtime)

                if creation_time > cutoff_time:
                    recent_models.append({
                        'path': model_dir,
                        'name': os.path.basename(model_dir),
                        'created': creation_time,
                        'size_mb': get_dir_size(model_dir) / (1024 * 1024)
                    })

    return sorted(recent_models, key=lambda x: x['created'], reverse=True)


def get_dir_size(path):
    """Get total size of directory in bytes."""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)
    except (OSError, FileNotFoundError):
        pass
    return total


def analyze_model_metadata_enhanced(model_dir):
    """Extract enhanced metadata from a model directory."""
    metadata_path = os.path.join(model_dir, "metadata.json")

    if not os.path.exists(metadata_path):
        return None

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Extract basic metadata
        result = {
            'version': metadata.get('version', 'unknown'),
            'timestamp': metadata.get('timestamp', 'unknown'),
            'flare_class': metadata.get('flare_class', 'unknown'),
            'time_window': metadata.get('time_window', 'unknown'),
            'description': metadata.get('description', ''),
            'performance': metadata.get('performance', {}),
            'hyperparameters': metadata.get('hyperparameters', {}),
            'training_metrics': metadata.get('training_metrics', {})
        }

        # Extract ablation-specific metadata
        ablation_metadata = metadata.get('ablation_metadata', {})
        if ablation_metadata:
            result.update({
                'is_ablation': True,
                'experiment_type': ablation_metadata.get('experiment_type', 'unknown'),
                'ablation_variant': ablation_metadata.get('variant', 'unknown'),
                'ablation_seed': ablation_metadata.get('seed', 'unknown'),
                'ablation_config': ablation_metadata.get('ablation_config', {}),
                'ablation_description': ablation_metadata.get('description', '')
            })
        else:
            # Check hyperparameters for ablation info (fallback)
            hyperparams = metadata.get('hyperparameters', {})
            if 'ablation_variant' in hyperparams:
                result.update({
                    'is_ablation': True,
                    'experiment_type': 'component_ablation',
                    'ablation_variant': hyperparams.get('ablation_variant', 'unknown'),
                    'ablation_seed': hyperparams.get('ablation_seed', 'unknown'),
                    'ablation_config': {
                        'use_attention_bottleneck': hyperparams.get('use_attention_bottleneck', True),
                        'use_evidential': hyperparams.get('use_evidential', True),
                        'use_evt': hyperparams.get('use_evt', True),
                        'use_precursor': hyperparams.get('use_precursor', True),
                        'loss_weights': hyperparams.get('loss_weights', {})
                    }
                })
            else:
                # Check description for ablation keywords
                desc = metadata.get('description', '').lower()
                if 'ablation' in desc:
                    result.update({
                        'is_ablation': True,
                        'experiment_type': 'possible_ablation',
                        'ablation_variant': 'unknown',
                        'ablation_seed': 'unknown'
                    })
                else:
                    result['is_ablation'] = False

        return result

    except (json.JSONDecodeError, FileNotFoundError):
        return None


def identify_ablation_models_enhanced(recent_models):
    """Identify ablation models using enhanced metadata."""
    ablation_models = []

    for model_info in recent_models:
        metadata = analyze_model_metadata_enhanced(model_info['path'])

        if metadata and metadata.get('is_ablation', False):
            model_info['metadata'] = metadata
            ablation_models.append(model_info)

    return ablation_models


def organize_ablation_results_enhanced(ablation_models):
    """Organize ablation results with enhanced metadata tracking."""

    # Create ablation results directories
    results_dir = Path("models/ablation/results")
    results_dir.mkdir(exist_ok=True)

    plots_dir = Path("models/ablation/plots")
    plots_dir.mkdir(exist_ok=True)

    trained_models_dir = Path("models/ablation/trained_models")
    trained_models_dir.mkdir(exist_ok=True)

    # Create enhanced summary data
    summary_data = []

    for model_info in ablation_models:
        metadata = model_info.get('metadata', {})

        summary_data.append({
            'model_name': model_info['name'],
            'model_path': model_info['path'],
            'version': metadata.get('version', 'unknown'),
            'created': model_info['created'].isoformat(),
            'size_mb': round(model_info['size_mb'], 2),
            'experiment_type': metadata.get('experiment_type', 'unknown'),
            'ablation_variant': metadata.get('ablation_variant', 'unknown'),
            'ablation_seed': metadata.get('ablation_seed', 'unknown'),
            'accuracy': metadata.get('performance', {}).get('accuracy', 'unknown'),
            'tss': metadata.get('performance', {}).get('TSS', 'unknown'),
            'roc_auc': metadata.get('performance', {}).get('ROC_AUC', 'unknown'),
            'brier': metadata.get('performance', {}).get('Brier', 'unknown'),
            'flare_class': metadata.get('flare_class', 'unknown'),
            'time_window': metadata.get('time_window', 'unknown'),
            'description': metadata.get('ablation_description', metadata.get('description', ''))
        })

        # Copy model to trained_models directory with descriptive name
        variant = metadata.get('ablation_variant', 'unknown')
        seed = metadata.get('ablation_seed', 'unknown')
        dest_name = f"{variant}_seed{seed}_{model_info['name']}"
        dest_path = trained_models_dir / dest_name

        if not dest_path.exists():
            try:
                shutil.copytree(model_info['path'], dest_path)
                print(f"✅ Copied {variant}_seed{seed} to {dest_path}")
            except Exception as e:
                print(f"❌ Failed to copy {model_info['name']}: {e}")

    # Save enhanced summary CSV
    if summary_data:
        df = pd.DataFrame(summary_data)
        summary_path = results_dir / "enhanced_ablation_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"✅ Saved enhanced summary to {summary_path}")

        # Create variant-wise summary
        if 'ablation_variant' in df.columns and df['ablation_variant'].notna().any():
            variant_summary = df.groupby(['ablation_variant', 'ablation_seed']).agg({
                'accuracy': 'first',
                'tss': 'first',
                'roc_auc': 'first',
                'brier': 'first',
                'created': 'first'
            }).reset_index()

            variant_path = results_dir / "variant_summary.csv"
            variant_summary.to_csv(variant_path, index=False)
            print(f"✅ Saved variant summary to {variant_path}")

            # Create pivot table for easy comparison
            if len(variant_summary) > 1:
                pivot_tss = variant_summary.pivot(index='ablation_variant', columns='ablation_seed', values='tss')
                pivot_acc = variant_summary.pivot(index='ablation_variant', columns='ablation_seed', values='accuracy')

                pivot_path = results_dir / "results_pivot.csv"
                with open(pivot_path, 'w') as f:
                    f.write("TSS Results by Variant and Seed\n")
                    pivot_tss.to_csv(f)
                    f.write("\n\nAccuracy Results by Variant and Seed\n")
                    pivot_acc.to_csv(f)
                print(f"✅ Saved pivot tables to {pivot_path}")

        return df
    else:
        print("❌ No ablation models found")
        return None


def print_ablation_summary(df):
    """Print a nice summary of ablation results."""
    if df is None or len(df) == 0:
        print("❌ No ablation results to summarize")
        return

    print(f"\n📊 ABLATION STUDY SUMMARY")
    print(f"=" * 60)
    print(f"Total experiments found: {len(df)}")

    if 'ablation_variant' in df.columns:
        variants = df['ablation_variant'].value_counts()
        print(f"\nExperiments by variant:")
        for variant, count in variants.items():
            print(f"   • {variant}: {count} experiments")

    if 'ablation_seed' in df.columns:
        seeds = df['ablation_seed'].value_counts()
        print(f"\nExperiments by seed:")
        for seed, count in seeds.items():
            print(f"   • Seed {seed}: {count} experiments")

    # Performance summary
    numeric_cols = ['accuracy', 'tss', 'roc_auc', 'brier']
    available_metrics = [col for col in numeric_cols if col in df.columns and df[col].notna().any()]

    if available_metrics:
        print(f"\n📈 Performance Summary:")
        for metric in available_metrics:
            values = pd.to_numeric(df[metric], errors='coerce').dropna()
            if len(values) > 0:
                print(f"   • {metric.upper()}: mean={values.mean():.4f}, std={values.std():.4f}, range=[{values.min():.4f}, {values.max():.4f}]")

    # Best performing experiments
    if 'tss' in df.columns:
        tss_values = pd.to_numeric(df['tss'], errors='coerce')
        if tss_values.notna().any():
            best_idx = tss_values.idxmax()
            best_exp = df.iloc[best_idx]
            print(f"\n🏆 Best TSS Performance:")
            print(f"   • Variant: {best_exp.get('ablation_variant', 'unknown')}")
            print(f"   • Seed: {best_exp.get('ablation_seed', 'unknown')}")
            print(f"   • TSS: {best_exp.get('tss', 'unknown')}")
            print(f"   • Accuracy: {best_exp.get('accuracy', 'unknown')}")


def main():
    """Main function to find and organize enhanced ablation results."""
    print("🔍 Searching for EVEREST ablation study results (Enhanced Metadata)...")
    print("=" * 80)

    # Find recent models
    recent_models = find_recent_everest_models(hours_back=24)
    print(f"📁 Found {len(recent_models)} recent EVEREST models")

    if not recent_models:
        print("❌ No recent models found. Check if ablation jobs completed successfully.")
        return

    # Show all recent models
    print("\n📋 Recent EVEREST models:")
    for model in recent_models:
        print(f"   • {model['name']} ({model['created'].strftime('%H:%M:%S')}, {model['size_mb']:.1f}MB)")

    # Identify ablation models using enhanced metadata
    ablation_models = identify_ablation_models_enhanced(recent_models)
    print(f"\n🎯 Identified {len(ablation_models)} ablation models with enhanced metadata")

    if ablation_models:
        print("\n🔬 Ablation models found:")
        for model in ablation_models:
            metadata = model.get('metadata', {})
            variant = metadata.get('ablation_variant', 'unknown')
            seed = metadata.get('ablation_seed', 'unknown')
            acc = metadata.get('performance', {}).get('accuracy', 'N/A')
            tss = metadata.get('performance', {}).get('TSS', 'N/A')
            exp_type = metadata.get('experiment_type', 'unknown')
            print(f"   • {variant}_seed{seed}: acc={acc}, tss={tss} ({exp_type})")

        # Organize results with enhanced metadata
        print(f"\n📂 Organizing enhanced ablation results...")
        summary_df = organize_ablation_results_enhanced(ablation_models)

        if summary_df is not None:
            print_ablation_summary(summary_df)

            print(f"\n✅ Enhanced results organized in:")
            print(f"   • models/ablation/results/enhanced_ablation_summary.csv")
            print(f"   • models/ablation/results/variant_summary.csv")
            print(f"   • models/ablation/results/results_pivot.csv")
            print(f"   • models/ablation/trained_models/ (copied models with variant_seed names)")
    else:
        print("❌ No ablation models identified")
        print("💡 This might mean:")
        print("   • Jobs are still running")
        print("   • Models saved without enhanced metadata")
        print("   • Check cluster logs for actual completion status")


if __name__ == "__main__":
    main()
