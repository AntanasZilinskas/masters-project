"""
Statistical Analysis for EVEREST Ablation Studies

This module implements statistical analysis including paired bootstrap tests,
confidence intervals, and significance testing as described in the paper.
"""

from .config import (
    ABLATION_VARIANTS,
    SEQUENCE_LENGTH_VARIANTS,
    RANDOM_SEEDS,
    EVALUATION_METRICS,
    STATISTICAL_CONFIG,
    OUTPUT_CONFIG,
)
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


class AblationAnalyzer:
    """
    Statistical analyzer for ablation study results.

    Implements:
    - Paired bootstrap tests (10,000 resamples)
    - 95% confidence intervals
    - Significance testing (p < 0.05)
    - Results aggregation across seeds
    """

    def __init__(self, results_dir: str = None):
        """Initialize analyzer with results directory."""
        self.results_dir = results_dir or OUTPUT_CONFIG["results_dir"]
        self.plots_dir = OUTPUT_CONFIG["plots_dir"]

        # Ensure output directories exist
        os.makedirs(self.plots_dir, exist_ok=True)

        self.results = {}
        self.aggregated_results = {}
        self.statistical_tests = {}

    def load_all_results(self):
        """Load all ablation study results."""
        print("📊 Loading ablation study results...")

        # Load component ablation results
        for variant_name in ABLATION_VARIANTS.keys():
            self.results[variant_name] = self._load_variant_results(variant_name)

        # Load sequence length ablation results
        for seq_variant in SEQUENCE_LENGTH_VARIANTS.keys():
            seq_key = f"sequence_{seq_variant}"
            self.results[seq_key] = self._load_sequence_results(seq_variant)

        print(f"✅ Loaded results for {len(self.results)} variants")

    def _load_variant_results(self, variant_name: str) -> List[Dict[str, Any]]:
        """Load results for a specific variant across all seeds."""
        variant_results = []

        for seed in RANDOM_SEEDS:
            experiment_name = f"ablation_{variant_name}_seed{seed}"
            result_file = os.path.join(
                self.results_dir, experiment_name, "results.json"
            )

            if os.path.exists(result_file):
                try:
                    with open(result_file, "r") as f:
                        result = json.load(f)
                    variant_results.append(result)
                except Exception as e:
                    print(f"⚠️ Failed to load {result_file}: {e}")
            else:
                print(f"⚠️ Missing result file: {result_file}")

        return variant_results

    def _load_sequence_results(self, seq_variant: str) -> List[Dict[str, Any]]:
        """Load results for a specific sequence length variant."""
        seq_results = []

        for seed in RANDOM_SEEDS:
            experiment_name = f"ablation_full_model_{seq_variant}_seed{seed}"
            result_file = os.path.join(
                self.results_dir, experiment_name, "results.json"
            )

            if os.path.exists(result_file):
                try:
                    with open(result_file, "r") as f:
                        result = json.load(f)
                    seq_results.append(result)
                except Exception as e:
                    print(f"⚠️ Failed to load {result_file}: {e}")

        return seq_results

    def aggregate_results(self):
        """Aggregate results across seeds for each variant."""
        print("📈 Aggregating results across seeds...")

        for variant_name, variant_results in self.results.items():
            if not variant_results:
                continue

            aggregated = {}

            # Extract metrics from all seeds
            metrics_by_seed = []
            for result in variant_results:
                if "final_metrics" in result:
                    metrics_by_seed.append(result["final_metrics"])

            if not metrics_by_seed:
                continue

            # Calculate statistics for each metric
            for metric in EVALUATION_METRICS:
                if metric in metrics_by_seed[0]:
                    values = [m[metric] for m in metrics_by_seed if metric in m]
                    if values:
                        aggregated[metric] = {
                            "mean": np.mean(values),
                            "std": np.std(values, ddof=1),
                            "values": values,
                            "n_seeds": len(values),
                        }

            self.aggregated_results[variant_name] = aggregated

        print(f"✅ Aggregated results for {len(self.aggregated_results)} variants")

    def perform_statistical_tests(self):
        """Perform paired bootstrap tests against full model baseline."""
        print("🔬 Performing statistical tests...")

        baseline_name = "full_model"
        if baseline_name not in self.aggregated_results:
            print(f"❌ Baseline '{baseline_name}' not found in results")
            return

        baseline_results = self.aggregated_results[baseline_name]

        for variant_name, variant_results in self.aggregated_results.items():
            if variant_name == baseline_name:
                continue

            self.statistical_tests[variant_name] = {}

            for metric in EVALUATION_METRICS:
                if metric in baseline_results and metric in variant_results:
                    baseline_values = baseline_results[metric]["values"]
                    variant_values = variant_results[metric]["values"]

                    # Perform paired bootstrap test
                    test_result = self._paired_bootstrap_test(
                        baseline_values, variant_values, metric
                    )

                    self.statistical_tests[variant_name][metric] = test_result

        print(
            f"✅ Completed statistical tests for {len(self.statistical_tests)} variants"
        )

    def _paired_bootstrap_test(
        self, baseline_values: List[float], variant_values: List[float], metric: str
    ) -> Dict[str, Any]:
        """Perform paired bootstrap test between baseline and variant."""

        if len(baseline_values) != len(variant_values):
            print(
                f"⚠️ Mismatched sample sizes for {metric}: {len(baseline_values)} vs {len(variant_values)}"
            )
            # Truncate to minimum length
            min_len = min(len(baseline_values), len(variant_values))
            baseline_values = baseline_values[:min_len]
            variant_values = variant_values[:min_len]

        # Calculate observed difference
        baseline_mean = np.mean(baseline_values)
        variant_mean = np.mean(variant_values)
        observed_diff = variant_mean - baseline_mean

        # Bootstrap resampling
        n_samples = len(baseline_values)
        n_bootstrap = STATISTICAL_CONFIG["bootstrap_samples"]
        bootstrap_diffs = []

        np.random.seed(42)  # For reproducibility

        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)

            baseline_resample = [baseline_values[i] for i in indices]
            variant_resample = [variant_values[i] for i in indices]

            diff = np.mean(variant_resample) - np.mean(baseline_resample)
            bootstrap_diffs.append(diff)

        bootstrap_diffs = np.array(bootstrap_diffs)

        # Calculate confidence interval
        alpha = 1 - STATISTICAL_CONFIG["confidence_level"]
        ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

        # Calculate p-value (two-tailed test)
        # H0: no difference, H1: there is a difference
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))

        # Determine significance
        is_significant = p_value < STATISTICAL_CONFIG["significance_threshold"]

        return {
            "observed_diff": observed_diff,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "is_significant": is_significant,
            "baseline_mean": baseline_mean,
            "variant_mean": variant_mean,
            "baseline_std": np.std(baseline_values, ddof=1),
            "variant_std": np.std(variant_values, ddof=1),
            "effect_size": observed_diff / np.std(baseline_values, ddof=1)
            if np.std(baseline_values, ddof=1) > 0
            else 0,
        }

    def generate_summary_table(self) -> pd.DataFrame:
        """Generate summary table similar to Table in the paper."""
        print("📋 Generating summary table...")

        baseline_name = "full_model"
        baseline_results = self.aggregated_results.get(baseline_name, {})

        # Define the order of variants as in the paper
        variant_order = [
            "full_model",
            "no_evidential",
            "no_evt",
            "mean_pool",
            "cross_entropy",
            "no_precursor",
            "fp32_training",
        ]

        # Create summary data
        summary_data = []

        for variant_name in variant_order:
            if variant_name not in self.aggregated_results:
                continue

            variant_results = self.aggregated_results[variant_name]
            variant_display_name = ABLATION_VARIANTS[variant_name]["name"]

            row = {"Variant": variant_display_name}

            # Add metrics with mean ± std format
            for metric in ["tss", "f1", "ece", "brier", "latency_ms"]:
                if metric in variant_results:
                    mean_val = variant_results[metric]["mean"]
                    std_val = variant_results[metric]["std"]

                    if variant_name == baseline_name:
                        # Baseline values (no delta)
                        if metric == "latency_ms":
                            row[metric] = f"{mean_val:.0f}"
                        elif metric in ["tss", "f1"]:
                            row[metric] = f"{mean_val:.3f}"
                        else:
                            row[metric] = f"{mean_val:.3f}"
                    else:
                        # Show delta from baseline
                        if (
                            variant_name in self.statistical_tests
                            and metric in self.statistical_tests[variant_name]
                        ):
                            test_result = self.statistical_tests[variant_name][metric]
                            delta = test_result["observed_diff"]
                            is_sig = test_result["is_significant"]

                            # Format delta with appropriate sign
                            if metric == "latency_ms":
                                delta_str = f"{delta:+.0f}"
                            else:
                                delta_str = f"{delta:+.3f}"

                            # Bold if significant
                            if is_sig:
                                row[metric] = f"**{delta_str}**"
                            else:
                                row[metric] = delta_str
                        else:
                            row[metric] = "N/A"

            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)

        # Rename columns to match paper
        column_mapping = {
            "tss": "TSS ↑",
            "f1": "F₁ ↑",
            "ece": "ECE ↓",
            "brier": "Brier ↓",
            "latency_ms": "Latency (ms)",
        }
        summary_df = summary_df.rename(columns=column_mapping)

        return summary_df

    def generate_sequence_length_table(self) -> pd.DataFrame:
        """Generate summary table for sequence length ablation."""
        print("📏 Generating sequence length table...")

        baseline_seq = "seq_10"  # Current baseline

        # Define order
        seq_order = ["seq_5", "seq_7", "seq_10", "seq_15", "seq_20"]

        summary_data = []

        for seq_variant in seq_order:
            seq_key = f"sequence_{seq_variant}"
            if seq_key not in self.aggregated_results:
                continue

            seq_results = self.aggregated_results[seq_key]
            seq_config = SEQUENCE_LENGTH_VARIANTS[seq_variant]

            row = {
                "Sequence Length": seq_config["input_shape"][0],
                "Description": seq_config["name"],
            }

            # Add metrics
            for metric in ["tss", "f1", "ece", "brier", "latency_ms"]:
                if metric in seq_results:
                    mean_val = seq_results[metric]["mean"]
                    std_val = seq_results[metric]["std"]

                    if metric == "latency_ms":
                        row[metric] = f"{mean_val:.1f} ± {std_val:.1f}"
                    else:
                        row[metric] = f"{mean_val:.3f} ± {std_val:.3f}"

            summary_data.append(row)

        return pd.DataFrame(summary_data)

    def plot_ablation_results(self):
        """Generate comprehensive plots for ablation results."""
        print("📊 Generating ablation plots...")

        # 1. Main ablation bar chart
        self._plot_ablation_bar_chart()

        # 2. Sequence length analysis
        self._plot_sequence_length_analysis()

        # 3. Statistical significance heatmap
        self._plot_significance_heatmap()

        # 4. Effect size analysis
        self._plot_effect_sizes()

        print(f"✅ Plots saved to {self.plots_dir}")

    def _plot_ablation_bar_chart(self):
        """Create bar chart showing TSS deltas for each ablation."""
        baseline_name = "full_model"
        baseline_tss = self.aggregated_results[baseline_name]["tss"]["mean"]

        # Prepare data
        variants = []
        deltas = []
        errors = []
        significance = []

        variant_order = [
            "no_evidential",
            "no_evt",
            "mean_pool",
            "cross_entropy",
            "no_precursor",
            "fp32_training",
        ]

        for variant_name in variant_order:
            if variant_name in self.statistical_tests:
                test_result = self.statistical_tests[variant_name].get("tss", {})
                if test_result:
                    variants.append(ABLATION_VARIANTS[variant_name]["name"])
                    deltas.append(test_result["observed_diff"])

                    # Error bars from confidence interval
                    ci_width = (test_result["ci_upper"] - test_result["ci_lower"]) / 2
                    errors.append(ci_width)

                    significance.append(test_result["is_significant"])

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Color bars by significance
        colors = ["red" if sig else "lightcoral" for sig in significance]

        bars = ax.bar(
            range(len(variants)),
            deltas,
            yerr=errors,
            color=colors,
            alpha=0.7,
            capsize=5,
        )

        # Customize plot
        ax.set_xlabel("Ablation Variant", fontsize=12)
        ax.set_ylabel("TSS Change from Full Model", fontsize=12)
        ax.set_title(
            "Impact of Component Ablations on TSS Performance",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=45, ha="right")
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)

        # Add significance indicators
        for i, (bar, sig) in enumerate(zip(bars, significance)):
            if sig:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + errors[i] + 0.005,
                    "*",
                    ha="center",
                    va="bottom",
                    fontsize=16,
                    fontweight="bold",
                )

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="red", alpha=0.7, label="Significant (p < 0.05)"),
            Patch(facecolor="lightcoral", alpha=0.7, label="Not significant"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, "ablation_tss_impact.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_sequence_length_analysis(self):
        """Plot sequence length analysis."""
        seq_lengths = []
        tss_means = []
        tss_stds = []
        latency_means = []
        latency_stds = []

        for seq_variant in ["seq_5", "seq_7", "seq_10", "seq_15", "seq_20"]:
            seq_key = f"sequence_{seq_variant}"
            if seq_key in self.aggregated_results:
                seq_config = SEQUENCE_LENGTH_VARIANTS[seq_variant]
                seq_results = self.aggregated_results[seq_key]

                seq_lengths.append(seq_config["input_shape"][0])

                if "tss" in seq_results:
                    tss_means.append(seq_results["tss"]["mean"])
                    tss_stds.append(seq_results["tss"]["std"])

                if "latency_ms" in seq_results:
                    latency_means.append(seq_results["latency_ms"]["mean"])
                    latency_stds.append(seq_results["latency_ms"]["std"])

        # Create dual-axis plot
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # TSS plot
        color1 = "tab:blue"
        ax1.set_xlabel("Sequence Length (timesteps)")
        ax1.set_ylabel("TSS", color=color1)
        ax1.errorbar(
            seq_lengths,
            tss_means,
            yerr=tss_stds,
            color=color1,
            marker="o",
            linewidth=2,
            markersize=8,
            capsize=5,
        )
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        # Latency plot
        ax2 = ax1.twinx()
        color2 = "tab:red"
        ax2.set_ylabel("Latency (ms)", color=color2)
        ax2.errorbar(
            seq_lengths,
            latency_means,
            yerr=latency_stds,
            color=color2,
            marker="s",
            linewidth=2,
            markersize=8,
            capsize=5,
        )
        ax2.tick_params(axis="y", labelcolor=color2)

        plt.title(
            "Sequence Length Impact on Performance and Latency",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, "sequence_length_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_significance_heatmap(self):
        """Create heatmap showing statistical significance across metrics."""
        # Prepare data for heatmap
        variants = []
        metrics = ["tss", "f1", "ece", "brier"]

        significance_matrix = []

        for variant_name in [
            "no_evidential",
            "no_evt",
            "mean_pool",
            "cross_entropy",
            "no_precursor",
            "fp32_training",
        ]:
            if variant_name in self.statistical_tests:
                variants.append(ABLATION_VARIANTS[variant_name]["name"])
                row = []
                for metric in metrics:
                    if metric in self.statistical_tests[variant_name]:
                        is_sig = self.statistical_tests[variant_name][metric][
                            "is_significant"
                        ]
                        row.append(1 if is_sig else 0)
                    else:
                        row.append(0)
                significance_matrix.append(row)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(significance_matrix, cmap="RdYlBu_r", aspect="auto")

        # Set ticks and labels
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels(variants)

        # Add text annotations
        for i in range(len(variants)):
            for j in range(len(metrics)):
                text = "✓" if significance_matrix[i][j] else "✗"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="white" if significance_matrix[i][j] else "black",
                    fontsize=14,
                )

        ax.set_title(
            "Statistical Significance of Ablation Effects\n(p < 0.05)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, "significance_heatmap.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_effect_sizes(self):
        """Plot effect sizes for different ablations."""
        variants = []
        effect_sizes = []

        for variant_name in [
            "no_evidential",
            "no_evt",
            "mean_pool",
            "cross_entropy",
            "no_precursor",
            "fp32_training",
        ]:
            if (
                variant_name in self.statistical_tests
                and "tss" in self.statistical_tests[variant_name]
            ):
                test_result = self.statistical_tests[variant_name]["tss"]
                variants.append(ABLATION_VARIANTS[variant_name]["name"])
                effect_sizes.append(abs(test_result["effect_size"]))

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(range(len(variants)), effect_sizes, color="steelblue", alpha=0.7)

        ax.set_xlabel("Ablation Variant")
        ax.set_ylabel("Effect Size (|Cohen's d|)")
        ax.set_title(
            "Effect Sizes of Component Ablations", fontsize=14, fontweight="bold"
        )
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=45, ha="right")

        # Add effect size interpretation lines
        ax.axhline(
            y=0.2, color="green", linestyle="--", alpha=0.7, label="Small effect"
        )
        ax.axhline(
            y=0.5, color="orange", linestyle="--", alpha=0.7, label="Medium effect"
        )
        ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.7, label="Large effect")

        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, "effect_sizes.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def save_results(self):
        """Save all analysis results."""
        print("💾 Saving analysis results...")

        # Save summary tables
        summary_table = self.generate_summary_table()
        summary_table.to_csv(
            os.path.join(self.plots_dir, "ablation_summary_table.csv"), index=False
        )

        # Save sequence length table
        seq_table = self.generate_sequence_length_table()
        seq_table.to_csv(
            os.path.join(self.plots_dir, "sequence_length_table.csv"), index=False
        )

        # Save statistical test results
        with open(os.path.join(self.plots_dir, "statistical_tests.json"), "w") as f:
            json.dump(self.statistical_tests, f, indent=2, default=str)

        # Save aggregated results
        with open(os.path.join(self.plots_dir, "aggregated_results.json"), "w") as f:
            json.dump(self.aggregated_results, f, indent=2, default=str)

        print(f"✅ Analysis results saved to {self.plots_dir}")

    def run_full_analysis(self):
        """Run complete ablation analysis pipeline."""
        print("🔬 Running full ablation analysis...")

        self.load_all_results()
        self.aggregate_results()
        self.perform_statistical_tests()
        self.plot_ablation_results()
        self.save_results()

        print("✅ Ablation analysis complete!")

        # Print summary
        print("\n📋 ABLATION STUDY SUMMARY")
        print("=" * 50)

        summary_table = self.generate_summary_table()
        print(summary_table.to_string(index=False))

        print(f"\n📊 Results saved to: {self.plots_dir}")


if __name__ == "__main__":
    analyzer = AblationAnalyzer()
    analyzer.run_full_analysis()
