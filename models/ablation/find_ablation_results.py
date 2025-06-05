#!/usr/bin/env python3
"""
Find and organize EVEREST ablation study results.

The ablation jobs save models using the standard EVEREST versioning system,
so we need to identify which models correspond to ablation experiments.
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

    # Look in both models/ and models/models/ directories
    search_patterns = ["models/EVEREST-v*", "EVEREST-v*"]

    recent_models = []

    for pattern in search_patterns:
        for model_dir in glob.glob(pattern):
            if os.path.isdir(model_dir):
                # Check creation time
                stat = os.stat(model_dir)
                creation_time = datetime.fromtimestamp(stat.st_mtime)

                if creation_time > cutoff_time:
                    recent_models.append(
                        {
                            "path": model_dir,
                            "name": os.path.basename(model_dir),
                            "created": creation_time,
                            "size_mb": get_dir_size(model_dir) / (1024 * 1024),
                        }
                    )

    return sorted(recent_models, key=lambda x: x["created"], reverse=True)


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


def analyze_model_metadata(model_dir):
    """Extract metadata from a model directory."""
    metadata_path = os.path.join(model_dir, "metadata.json")

    if not os.path.exists(metadata_path):
        return None

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        return {
            "version": metadata.get("version", "unknown"),
            "timestamp": metadata.get("timestamp", "unknown"),
            "flare_class": metadata.get("flare_class", "unknown"),
            "time_window": metadata.get("time_window", "unknown"),
            "description": metadata.get("description", ""),
            "performance": metadata.get("performance", {}),
            "hyperparameters": metadata.get("hyperparameters", {}),
            "training_metrics": metadata.get("training_metrics", {}),
        }
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def identify_ablation_models(recent_models):
    """Identify which recent models are from ablation experiments."""
    ablation_models = []

    for model_info in recent_models:
        metadata = analyze_model_metadata(model_info["path"])

        if metadata:
            # Check if this looks like an ablation experiment
            # Ablation models should be M5-class, 72h window, recent timestamp
            if (
                metadata["flare_class"] == "M5"
                and metadata["time_window"] == "72"
                and "ablation" in metadata.get("description", "").lower()
            ):
                model_info["metadata"] = metadata
                model_info["experiment_type"] = "ablation"
                ablation_models.append(model_info)
            elif metadata["flare_class"] == "M5" and metadata["time_window"] == "72":
                # Might be ablation even without explicit description
                model_info["metadata"] = metadata
                model_info["experiment_type"] = "possible_ablation"
                ablation_models.append(model_info)

    return ablation_models


def organize_ablation_results(ablation_models):
    """Organize ablation results into the expected directory structure."""

    # Create ablation results directories
    results_dir = Path("models/ablation/results")
    results_dir.mkdir(exist_ok=True)

    plots_dir = Path("models/ablation/plots")
    plots_dir.mkdir(exist_ok=True)

    trained_models_dir = Path("models/ablation/trained_models")
    trained_models_dir.mkdir(exist_ok=True)

    # Create summary data
    summary_data = []

    for model_info in ablation_models:
        metadata = model_info.get("metadata", {})

        # Extract experiment details from model name or metadata
        model_name = model_info["name"]

        # Try to infer variant and seed from timing or version
        summary_data.append(
            {
                "model_name": model_name,
                "model_path": model_info["path"],
                "version": metadata.get("version", "unknown"),
                "created": model_info["created"].isoformat(),
                "size_mb": round(model_info["size_mb"], 2),
                "accuracy": metadata.get("performance", {}).get("accuracy", "unknown"),
                "tss": metadata.get("performance", {}).get("TSS", "unknown"),
                "flare_class": metadata.get("flare_class", "unknown"),
                "time_window": metadata.get("time_window", "unknown"),
                "experiment_type": model_info.get("experiment_type", "unknown"),
            }
        )

        # Copy model to trained_models directory with descriptive name
        dest_name = f"{model_name}_ablation"
        dest_path = trained_models_dir / dest_name

        if not dest_path.exists():
            try:
                shutil.copytree(model_info["path"], dest_path)
                print(f"✅ Copied {model_name} to {dest_path}")
            except Exception as e:
                print(f"❌ Failed to copy {model_name}: {e}")

    # Save summary CSV
    if summary_data:
        df = pd.DataFrame(summary_data)
        summary_path = results_dir / "ablation_models_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"✅ Saved summary to {summary_path}")

        return df
    else:
        print("❌ No ablation models found")
        return None


def main():
    """Main function to find and organize ablation results."""
    print("🔍 Searching for EVEREST ablation study results...")
    print("=" * 60)

    # Find recent models
    recent_models = find_recent_everest_models(hours_back=24)
    print(f"📁 Found {len(recent_models)} recent EVEREST models")

    if not recent_models:
        print(
            "❌ No recent models found. Check if ablation jobs completed successfully."
        )
        return

    # Show all recent models
    print("\n📋 Recent EVEREST models:")
    for model in recent_models:
        print(
            f"   • {model['name']} ({model['created'].strftime('%H:%M:%S')}, {model['size_mb']:.1f}MB)"
        )

    # Identify ablation models
    ablation_models = identify_ablation_models(recent_models)
    print(f"\n🎯 Identified {len(ablation_models)} potential ablation models")

    if ablation_models:
        print("\n🔬 Ablation models found:")
        for model in ablation_models:
            metadata = model.get("metadata", {})
            acc = metadata.get("performance", {}).get("accuracy", "N/A")
            tss = metadata.get("performance", {}).get("TSS", "N/A")
            print(
                f"   • {model['name']}: acc={acc}, tss={tss} ({model['experiment_type']})"
            )

        # Organize results
        print(f"\n📂 Organizing ablation results...")
        summary_df = organize_ablation_results(ablation_models)

        if summary_df is not None:
            print(f"\n📊 Ablation Results Summary:")
            print(summary_df.to_string(index=False))

            print(f"\n✅ Results organized in:")
            print(f"   • models/ablation/results/ablation_models_summary.csv")
            print(f"   • models/ablation/trained_models/ (copied models)")
    else:
        print("❌ No ablation models identified")
        print("💡 This might mean:")
        print("   • Jobs are still running")
        print("   • Models saved with different naming")
        print("   • Check cluster logs for actual completion status")


if __name__ == "__main__":
    main()
