#!/usr/bin/env python3
"""
Publication Results Generator for EVEREST Ablation Study

Generates LaTeX-ready tables and statistics for publication, including:
- Component ablation tables with TSS, std, params, and p-values
- Bootstrap statistical significance testing
- Publication-quality formatting
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.utils import resample
import warnings

warnings.filterwarnings("ignore")


def find_ablation_models_for_publication():
    """Find all ablation models with enhanced metadata for publication analysis."""

    # Search patterns prioritizing models/models/
    search_patterns = [
        "../../models/models/EVEREST-v*",
        "../../models/EVEREST-v*",
        "../models/models/EVEREST-v*",
        "../models/EVEREST-v*",
    ]

    ablation_models = []

    for pattern in search_patterns:
        for model_dir in glob.glob(pattern):
            if os.path.isdir(model_dir):
                metadata_path = os.path.join(model_dir, "metadata.json")

                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)

                        # Check for enhanced ablation metadata
                        ablation_metadata = metadata.get("ablation_metadata", {})
                        hyperparams = metadata.get("hyperparameters", {})

                        is_ablation = False
                        variant = "unknown"
                        seed = "unknown"

                        if ablation_metadata:
                            # Enhanced metadata format
                            is_ablation = True
                            variant = ablation_metadata.get("variant", "unknown")
                            seed = ablation_metadata.get("seed", "unknown")
                        elif "ablation_variant" in hyperparams:
                            # Fallback to hyperparameters
                            is_ablation = True
                            variant = hyperparams.get("ablation_variant", "unknown")
                            seed = hyperparams.get("ablation_seed", "unknown")
                        elif "ablation" in metadata.get("description", "").lower():
                            # Last resort: description parsing
                            is_ablation = True
                            # Try to extract variant from description
                            desc = metadata.get("description", "").lower()
                            for v in [
                                "full_model",
                                "no_evidential",
                                "no_evt",
                                "mean_pool",
                                "cross_entropy",
                                "no_precursor",
                                "fp32_training",
                            ]:
                                if v in desc:
                                    variant = v
                                    break

                        if is_ablation:
                            performance = metadata.get("performance", {})
                            architecture = metadata.get("architecture", {})

                            ablation_models.append(
                                {
                                    "model_dir": model_dir,
                                    "model_name": os.path.basename(model_dir),
                                    "variant": variant,
                                    "seed": seed,
                                    "tss": performance.get(
                                        "TSS", performance.get("tss", np.nan)
                                    ),
                                    "accuracy": performance.get("accuracy", np.nan),
                                    "roc_auc": performance.get(
                                        "ROC_AUC", performance.get("roc_auc", np.nan)
                                    ),
                                    "brier": performance.get(
                                        "Brier", performance.get("brier", np.nan)
                                    ),
                                    "ece": performance.get(
                                        "ECE", performance.get("ece", np.nan)
                                    ),
                                    "num_params": architecture.get(
                                        "num_params", "unknown"
                                    ),
                                    "flare_class": metadata.get(
                                        "flare_class", "unknown"
                                    ),
                                    "time_window": metadata.get(
                                        "time_window", "unknown"
                                    ),
                                    "metadata": metadata,
                                }
                            )

                    except (json.JSONDecodeError, Exception) as e:
                        continue

    return ablation_models


def calculate_parameter_differences():
    """Calculate parameter differences for each ablation variant."""

    # Baseline parameter counts (approximate for EVEREST architecture)
    # These should be updated with actual values from your models
    baseline_params = {
        "embedding": 9 * 64,  # input_dim * embed_dim
        "transformers": 8
        * (64 * 64 * 4 + 64 * 256 * 2),  # num_blocks * (attention + ffn)
        "attention_bottleneck": 64 * 1,  # embed_dim * 1 for attention pooling
        "evidential_head": 128 * 4,  # hidden_dim * 4 (mu, v, alpha, beta)
        "evt_head": 128 * 2,  # hidden_dim * 2 (xi, sigma)
        "precursor_head": 128 * 1,  # hidden_dim * 1
        "main_head": 128 * 1,  # hidden_dim * 1
    }

    # Calculate differences for each variant
    param_diffs = {
        "full_model": 0,  # baseline
        "no_evidential": -baseline_params["evidential_head"],
        "no_evt": -baseline_params["evt_head"],
        "mean_pool": -baseline_params["attention_bottleneck"],
        "cross_entropy": -(
            baseline_params["evidential_head"] + baseline_params["evt_head"]
        ),
        "no_precursor": -baseline_params["precursor_head"],
        "fp32_training": 0,  # same architecture, different precision
    }

    return param_diffs


def bootstrap_significance_test(baseline_scores, variant_scores, n_bootstrap=10000):
    """
    Perform bootstrap significance test between baseline and variant.

    Returns p-value for the hypothesis that variant is significantly different from baseline.
    """
    if len(baseline_scores) == 0 or len(variant_scores) == 0:
        return np.nan

    # Observed difference
    obs_diff = np.mean(variant_scores) - np.mean(baseline_scores)

    # Bootstrap resampling
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        boot_baseline = resample(baseline_scores, n_samples=len(baseline_scores))
        boot_variant = resample(variant_scores, n_samples=len(variant_scores))

        # Calculate difference
        boot_diff = np.mean(boot_variant) - np.mean(boot_baseline)
        bootstrap_diffs.append(boot_diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Two-tailed p-value
    if obs_diff >= 0:
        p_value = 2 * np.mean(bootstrap_diffs <= -abs(obs_diff))
    else:
        p_value = 2 * np.mean(bootstrap_diffs >= abs(obs_diff))

    return min(p_value, 1.0)


def generate_component_ablation_table(ablation_models):
    """Generate LaTeX table for component ablation results."""

    if not ablation_models:
        print("❌ No ablation models found for publication analysis")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(ablation_models)

    # Filter for valid TSS scores
    df = df[df["tss"].notna() & (df["tss"] != "unknown")]
    df["tss"] = pd.to_numeric(df["tss"], errors="coerce")
    df = df.dropna(subset=["tss"])

    if len(df) == 0:
        print("❌ No valid TSS scores found in ablation models")
        return None

    print(f"📊 Found {len(df)} ablation experiments with valid TSS scores")
    print(f"   Variants: {df['variant'].unique()}")
    print(f"   Seeds per variant: {df.groupby('variant').size().to_dict()}")

    # Group by variant and calculate statistics
    variant_stats = []
    param_diffs = calculate_parameter_differences()

    # Get baseline (full_model) scores for significance testing
    baseline_df = df[df["variant"] == "full_model"]
    baseline_tss = baseline_df["tss"].values if len(baseline_df) > 0 else []

    for variant in [
        "full_model",
        "no_evidential",
        "no_evt",
        "mean_pool",
        "cross_entropy",
        "no_precursor",
        "fp32_training",
    ]:
        variant_df = df[df["variant"] == variant]

        if len(variant_df) == 0:
            continue

        tss_scores = variant_df["tss"].values

        # Calculate statistics
        mean_tss = np.mean(tss_scores)
        std_tss = np.std(tss_scores, ddof=1) if len(tss_scores) > 1 else 0.0

        # Parameter difference
        param_diff = param_diffs.get(variant, 0)

        # Statistical significance vs baseline
        if variant == "full_model":
            p_value = np.nan  # baseline
        else:
            p_value = bootstrap_significance_test(baseline_tss, tss_scores)

        # Variant display name mapping
        display_names = {
            "full_model": "Full Model (Baseline)",
            "no_evidential": "No Evidential Head",
            "no_evt": "No EVT Head",
            "mean_pool": "Mean Pooling",
            "cross_entropy": "Cross-Entropy Loss",
            "no_precursor": "No Precursor Head",
            "fp32_training": "FP32 Training",
        }

        variant_stats.append(
            {
                "variant": variant,
                "display_name": display_names.get(variant, variant),
                "mean_tss": mean_tss,
                "std_tss": std_tss,
                "param_diff": param_diff,
                "p_value": p_value,
                "n_seeds": len(tss_scores),
                "tss_scores": tss_scores,
            }
        )

    return variant_stats


def format_latex_table(variant_stats):
    """Format results as LaTeX table."""

    if not variant_stats:
        return "No data available for LaTeX table generation."

    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("  \\centering")
    latex_lines.append(
        "  \\caption{Component ablation study results on the validation set"
    )
    latex_lines.append("           (mean $\\pm$~s.d.\\ over five seeds).}")
    latex_lines.append("  \\label{tab:component_ablation}")
    latex_lines.append("  \\begin{tabular}{lcccc}")
    latex_lines.append("  \\toprule")
    latex_lines.append(
        "  Component & $\\Delta$ Params & TSS $\\uparrow$ & s.d. & $p$ \\\\"
    )
    latex_lines.append("  \\midrule")

    for stats in variant_stats:
        variant = stats["display_name"]
        param_diff = stats["param_diff"]
        mean_tss = stats["mean_tss"]
        std_tss = stats["std_tss"]
        p_value = stats["p_value"]

        # Format parameter difference
        if param_diff == 0:
            param_str = "—"
        elif param_diff > 0:
            param_str = (
                f"+{param_diff//1000:.0f}k" if param_diff >= 1000 else f"+{param_diff}"
            )
        else:
            param_str = (
                f"{param_diff//1000:.0f}k"
                if abs(param_diff) >= 1000
                else f"{param_diff}"
            )

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
        if stats["variant"] == "full_model":
            variant = f"\\textbf{{{variant}}}"

        latex_lines.append(
            f"  {variant:<25} & {param_str:>8} & {tss_str:>6} & {std_str:>6} & {p_str:>8} \\\\"
        )

    latex_lines.append("  \\bottomrule")
    latex_lines.append("  \\end{tabular}")
    latex_lines.append("\\end{table}")

    return "\n".join(latex_lines)


def generate_summary_statistics(variant_stats):
    """Generate comprehensive summary statistics."""

    if not variant_stats:
        return "No data available for summary statistics."

    summary_lines = []
    summary_lines.append("📊 PUBLICATION ABLATION STUDY SUMMARY")
    summary_lines.append("=" * 80)

    # Overall statistics
    total_experiments = sum(stats["n_seeds"] for stats in variant_stats)
    n_variants = len(variant_stats)

    summary_lines.append(f"Total experiments: {total_experiments}")
    summary_lines.append(f"Number of variants: {n_variants}")
    summary_lines.append(
        f"Seeds per variant: {variant_stats[0]['n_seeds'] if variant_stats else 'N/A'}"
    )
    summary_lines.append("")

    # Performance by variant
    summary_lines.append("📈 Performance by Variant:")
    for stats in variant_stats:
        variant = stats["display_name"]
        mean_tss = stats["mean_tss"]
        std_tss = stats["std_tss"]
        p_value = stats["p_value"]

        p_str = "baseline" if np.isnan(p_value) else f"p={p_value:.3f}"
        summary_lines.append(
            f"   • {variant:<25}: TSS={mean_tss:.4f}±{std_tss:.4f} ({p_str})"
        )

    summary_lines.append("")

    # Statistical significance summary
    significant_variants = [
        s for s in variant_stats if not np.isnan(s["p_value"]) and s["p_value"] < 0.05
    ]
    summary_lines.append(
        f"🎯 Statistically Significant Differences (p < 0.05): {len(significant_variants)}"
    )

    for stats in significant_variants:
        variant = stats["display_name"]
        p_value = stats["p_value"]
        mean_tss = stats["mean_tss"]
        baseline_tss = next(
            (s["mean_tss"] for s in variant_stats if s["variant"] == "full_model"),
            np.nan,
        )

        if not np.isnan(baseline_tss):
            diff = mean_tss - baseline_tss
            direction = "↑" if diff > 0 else "↓"
            summary_lines.append(
                f"   • {variant}: {direction} {abs(diff):.4f} TSS (p={p_value:.3f})"
            )

    # Best and worst performers
    summary_lines.append("")
    best_variant = max(variant_stats, key=lambda x: x["mean_tss"])
    worst_variant = min(variant_stats, key=lambda x: x["mean_tss"])

    summary_lines.append(
        f"🏆 Best Performance: {best_variant['display_name']} (TSS={best_variant['mean_tss']:.4f})"
    )
    summary_lines.append(
        f"📉 Worst Performance: {worst_variant['display_name']} (TSS={worst_variant['mean_tss']:.4f})"
    )

    return "\n".join(summary_lines)


def save_results_to_files(variant_stats, latex_table, summary_stats):
    """Save all results to organized files."""

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save LaTeX table
    latex_file = results_dir / "component_ablation_table.tex"
    with open(latex_file, "w") as f:
        f.write(latex_table)
    print(f"✅ LaTeX table saved to: {latex_file}")

    # Save summary statistics
    summary_file = results_dir / "ablation_summary.txt"
    with open(summary_file, "w") as f:
        f.write(summary_stats)
    print(f"✅ Summary statistics saved to: {summary_file}")

    # Save detailed CSV
    if variant_stats:
        csv_data = []
        for stats in variant_stats:
            for i, tss in enumerate(stats["tss_scores"]):
                csv_data.append(
                    {
                        "variant": stats["variant"],
                        "display_name": stats["display_name"],
                        "seed": i,
                        "tss": tss,
                        "mean_tss": stats["mean_tss"],
                        "std_tss": stats["std_tss"],
                        "param_diff": stats["param_diff"],
                        "p_value": stats["p_value"],
                    }
                )

        csv_df = pd.DataFrame(csv_data)
        csv_file = results_dir / "detailed_ablation_results.csv"
        csv_df.to_csv(csv_file, index=False)
        print(f"✅ Detailed CSV saved to: {csv_file}")

    # Save publication-ready statistics
    pub_stats_file = results_dir / "publication_statistics.txt"
    with open(pub_stats_file, "w") as f:
        f.write("PUBLICATION-READY STATISTICS\n")
        f.write("=" * 50 + "\n\n")

        if variant_stats:
            baseline_stats = next(
                (s for s in variant_stats if s["variant"] == "full_model"), None
            )
            if baseline_stats:
                f.write(
                    f"Baseline (Full Model): TSS = {baseline_stats['mean_tss']:.4f} ± {baseline_stats['std_tss']:.4f}\n\n"
                )

            f.write("Component Contributions:\n")
            for stats in variant_stats:
                if stats["variant"] != "full_model":
                    f.write(
                        f"- {stats['display_name']}: TSS = {stats['mean_tss']:.4f} ± {stats['std_tss']:.4f}"
                    )
                    if not np.isnan(stats["p_value"]):
                        f.write(f" (p = {stats['p_value']:.3f})")
                    f.write("\n")

    print(f"✅ Publication statistics saved to: {pub_stats_file}")


def main():
    """Main function to generate publication results."""

    print("🔬 EVEREST Ablation Study - Publication Results Generator")
    print("=" * 80)

    # Find ablation models
    print("📁 Searching for ablation models with enhanced metadata...")
    ablation_models = find_ablation_models_for_publication()

    if not ablation_models:
        print("❌ No ablation models found!")
        print("💡 Make sure you have run the enhanced ablation study first:")
        print("   cd models/ablation/cluster")
        print("   qsub submit_component_ablation_metadata.pbs")
        return

    print(f"✅ Found {len(ablation_models)} ablation experiments")

    # Generate component ablation table
    print("\n📊 Generating component ablation statistics...")
    variant_stats = generate_component_ablation_table(ablation_models)

    if not variant_stats:
        print("❌ Could not generate variant statistics")
        return

    # Generate LaTeX table
    print("📝 Formatting LaTeX table...")
    latex_table = format_latex_table(variant_stats)

    # Generate summary statistics
    print("📈 Generating summary statistics...")
    summary_stats = generate_summary_statistics(variant_stats)

    # Save all results
    print("\n💾 Saving results...")
    save_results_to_files(variant_stats, latex_table, summary_stats)

    # Display results
    print("\n" + summary_stats)
    print("\n📋 LaTeX Table:")
    print("-" * 50)
    print(latex_table)

    print(f"\n✅ Publication results generated successfully!")
    print(f"📁 Check the 'results/' directory for all output files")


if __name__ == "__main__":
    main()
