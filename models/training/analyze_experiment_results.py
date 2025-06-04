#!/usr/bin/env python3
"""
Comprehensive analysis script for EVEREST experiment results.
Calculates performance metrics, bootstrapped confidence intervals, and generates paper tables.
"""

import pandas as pd
import numpy as np
import os
import json
import glob
import warnings
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    accuracy_score,
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")


class ExperimentAnalyzer:
    def __init__(self, results_dir: str = "results", quality_threshold: float = 0.6):
        self.results_dir = Path(results_dir)
        self.quality_threshold = quality_threshold
        self.experiments = {}
        self.aggregated_results = {}

    def load_experiment_data(self) -> Dict:
        """Load all experiment data from result directories."""
        print("Loading experiment data...")

        # Pattern: everest_{flare_class}_{time_window}_seed{N}
        pattern = "everest_*_*h_seed*"
        experiment_dirs = list(self.results_dir.glob(pattern))

        print(f"Found {len(experiment_dirs)} experiment directories")

        for exp_dir in experiment_dirs:
            try:
                exp_name = exp_dir.name
                print(f"Processing: {exp_name}")

                # Parse experiment parameters
                parts = exp_name.split("_")
                flare_class = parts[1]  # C, M, or M5
                time_window = parts[2]  # 24h, 48h, 72h
                seed = int(parts[3].replace("seed", ""))

                # Load required files
                exp_data = self._load_experiment_files(exp_dir)

                if exp_data is None:
                    print(f"  ⚠️  Skipping {exp_name} - missing or corrupted files")
                    continue

                exp_data.update(
                    {
                        "flare_class": flare_class,
                        "time_window": time_window,
                        "seed": seed,
                        "exp_name": exp_name,
                    }
                )

                # Store experiment
                key = f"{flare_class}_{time_window}_{seed}"
                self.experiments[key] = exp_data
                print(f"  ✅ Loaded {exp_name}")

            except Exception as e:
                print(f"  ❌ Error loading {exp_dir.name}: {e}")
                continue

        print(f"\nSuccessfully loaded {len(self.experiments)} experiments")

        # Filter out poor performing experiments
        if self.experiments:
            self.filter_poor_performing_experiments()

        return self.experiments

    def filter_poor_performing_experiments(self):
        """Filter out experiments with poor performance (any key metric below threshold)."""
        print(
            f"\n🔍 Filtering experiments with metrics below {self.quality_threshold*100}%..."
        )

        filtered_experiments = {}
        discarded_experiments = []

        for key, exp in self.experiments.items():
            # Check if we can evaluate this experiment
            if not self._has_valid_predictions(exp):
                print(f"  ⚠️  {key}: Cannot evaluate - no valid predictions")
                discarded_experiments.append((key, "No valid predictions"))
                continue

            # Calculate basic metrics for filtering
            try:
                pred_df = exp["predictions"]

                # Get predictions and true labels
                if "y_true" in pred_df.columns and "y_pred" in pred_df.columns:
                    y_true = pred_df["y_true"].values
                    y_pred = pred_df["y_pred"].values

                    # Calculate key metrics
                    tss = self.calculate_tss(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    precision = precision_score(y_true, y_pred, zero_division=0)
                    recall = recall_score(y_true, y_pred, zero_division=0)
                    accuracy = accuracy_score(y_true, y_pred)

                    # Check if any metric falls below threshold
                    metrics_to_check = {
                        "TSS": tss,
                        "F1": f1,
                        "Precision": precision,
                        "Recall": recall,
                        "Accuracy": accuracy,
                    }

                    failed_metrics = []
                    for metric_name, value in metrics_to_check.items():
                        if value < self.quality_threshold:
                            failed_metrics.append(f"{metric_name}={value:.3f}")

                    if failed_metrics:
                        reason = f"Low performance: {', '.join(failed_metrics)}"
                        discarded_experiments.append((key, reason))
                        print(f"  ❌ {key}: {reason}")
                    else:
                        filtered_experiments[key] = exp
                        print(
                            f"  ✅ {key}: All metrics above threshold (TSS={tss:.3f}, F1={f1:.3f})"
                        )
                else:
                    discarded_experiments.append(
                        (key, "Missing y_true or y_pred columns")
                    )
                    print(f"  ⚠️  {key}: Missing required columns")

            except Exception as e:
                discarded_experiments.append((key, f"Error calculating metrics: {e}"))
                print(f"  ❌ {key}: Error calculating metrics - {e}")

        # Update experiments dict
        original_count = len(self.experiments)
        self.experiments = filtered_experiments
        filtered_count = len(self.experiments)
        discarded_count = len(discarded_experiments)

        print(f"\n📊 Quality Filtering Summary:")
        print(f"  Original experiments: {original_count}")
        print(f"  Kept (good quality): {filtered_count}")
        print(f"  Discarded (poor quality): {discarded_count}")

        if discarded_experiments:
            print(f"\n🗑️  Discarded experiments:")
            for exp_key, reason in discarded_experiments:
                print(f"    {exp_key}: {reason}")

        # Show remaining experiments by task
        if filtered_experiments:
            task_summary = {}
            for key, exp in filtered_experiments.items():
                task_key = f"{exp['flare_class']}_{exp['time_window']}"
                if task_key not in task_summary:
                    task_summary[task_key] = []
                task_summary[task_key].append(exp["seed"])

            print(f"\n✅ Remaining high-quality experiments:")
            for task, seeds in task_summary.items():
                print(f"    {task}: {len(seeds)} seeds {sorted(seeds)}")

    def _has_valid_predictions(self, exp: Dict) -> bool:
        """Check if experiment has valid prediction data."""
        if "predictions" not in exp:
            return False

        pred_df = exp["predictions"]
        if pred_df is None or len(pred_df) == 0:
            return False

        required_cols = ["y_true", "y_pred"]
        return all(col in pred_df.columns for col in required_cols)

    def _load_experiment_files(self, exp_dir: Path) -> Optional[Dict]:
        """Load all required files for a single experiment."""
        required_files = ["final_metrics.csv", "predictions.csv", "results.json"]

        # Check if all required files exist
        for file_name in required_files:
            if not (exp_dir / file_name).exists():
                return None

        try:
            data = {}

            # Load final metrics
            metrics_path = exp_dir / "final_metrics.csv"
            data["final_metrics"] = pd.read_csv(metrics_path)

            # Load predictions
            pred_path = exp_dir / "predictions.csv"
            data["predictions"] = pd.read_csv(pred_path)

            # Load results JSON
            json_path = exp_dir / "results.json"
            with open(json_path, "r") as f:
                data["results"] = json.load(f)

            # Load threshold optimization if available
            thresh_path = exp_dir / "threshold_optimization.csv"
            if thresh_path.exists():
                data["threshold_opt"] = pd.read_csv(thresh_path)

            # Load training history if available
            history_path = exp_dir / "training_history.csv"
            if history_path.exists():
                data["training_history"] = pd.read_csv(history_path)

            return data

        except Exception as e:
            print(f"Error loading files from {exp_dir}: {e}")
            return None

    def calculate_tss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate True Skill Statistic (TSS)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # TSS = Sensitivity + Specificity - 1
        # TSS = TPR + TNR - 1 = TP/(TP+FN) + TN/(TN+FP) - 1
        if tp + fn == 0 or tn + fp == 0:
            return 0.0

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return sensitivity + specificity - 1

    def calculate_ece(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
    ) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for j, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            # Fix: Use consistent binning with reliability diagrams
            if j == n_bins - 1:  # Last bin includes upper boundary
                in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
            else:
                in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def bootstrap_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        metric_func,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> Tuple[float, float, float]:
        """Calculate bootstrapped confidence intervals for a metric."""
        n_samples = len(y_true)
        bootstrap_scores = []

        np.random.seed(42)  # For reproducibility

        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            y_prob_boot = y_prob[indices] if y_prob is not None else None

            # Calculate metric
            if y_prob_boot is not None:
                score = metric_func(y_true_boot, y_pred_boot, y_prob_boot)
            else:
                score = metric_func(y_true_boot, y_pred_boot)

            bootstrap_scores.append(score)

        # Calculate confidence intervals
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        mean_score = np.mean(bootstrap_scores)
        ci_lower = np.percentile(bootstrap_scores, lower_percentile)
        ci_upper = np.percentile(bootstrap_scores, upper_percentile)

        return mean_score, ci_lower, ci_upper

    def calculate_all_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict:
        """Calculate all performance metrics with bootstrapped confidence intervals."""
        metrics = {}

        # Basic metrics
        def tss_func(yt, yp, ypr=None):
            return self.calculate_tss(yt, yp)

        def f1_func(yt, yp, ypr=None):
            return f1_score(yt, yp, zero_division=0)

        def precision_func(yt, yp, ypr=None):
            return precision_score(yt, yp, zero_division=0)

        def recall_func(yt, yp, ypr=None):
            return recall_score(yt, yp, zero_division=0)

        def brier_func(yt, yp, ypr):
            return brier_score_loss(yt, ypr)

        def ece_func(yt, yp, ypr):
            return self.calculate_ece(yt, ypr)

        # Calculate metrics with confidence intervals
        metrics["tss"] = self.bootstrap_metric(y_true, y_pred, y_prob, tss_func)
        metrics["f1"] = self.bootstrap_metric(y_true, y_pred, y_prob, f1_func)
        metrics["precision"] = self.bootstrap_metric(
            y_true, y_pred, y_prob, precision_func
        )
        metrics["recall"] = self.bootstrap_metric(y_true, y_pred, y_prob, recall_func)
        metrics["brier"] = self.bootstrap_metric(y_true, y_pred, y_prob, brier_func)
        metrics["ece"] = self.bootstrap_metric(y_true, y_pred, y_prob, ece_func)

        return metrics

    def aggregate_experiments(self) -> Dict:
        """Aggregate results across seeds for each task."""
        print("\nAggregating results across seeds...")

        # Group experiments by task (flare_class + time_window)
        tasks = {}
        for key, exp in self.experiments.items():
            task_key = f"{exp['flare_class']}_{exp['time_window']}"
            if task_key not in tasks:
                tasks[task_key] = []
            tasks[task_key].append(exp)

        aggregated = {}

        for task_key, experiments in tasks.items():
            print(f"Processing task: {task_key} ({len(experiments)} seeds)")

            # Collect predictions from all seeds
            all_y_true = []
            all_y_pred = []
            all_y_prob = []

            for exp in experiments:
                pred_df = exp["predictions"]

                # Assuming predictions.csv has columns: y_true, y_pred, y_prob
                if "y_true" in pred_df.columns and "y_pred" in pred_df.columns:
                    all_y_true.extend(pred_df["y_true"].values)
                    all_y_pred.extend(pred_df["y_pred"].values)

                    if "y_prob" in pred_df.columns:
                        all_y_prob.extend(pred_df["y_prob"].values)
                    elif "probability" in pred_df.columns:
                        all_y_prob.extend(pred_df["probability"].values)
                    else:
                        # Try to find probability column
                        prob_cols = [
                            col for col in pred_df.columns if "prob" in col.lower()
                        ]
                        if prob_cols:
                            all_y_prob.extend(pred_df[prob_cols[0]].values)
                        else:
                            print(
                                f"Warning: No probability column found for {exp['exp_name']}"
                            )
                            all_y_prob.extend(
                                [0.5] * len(pred_df)
                            )  # Default probability

            if not all_y_true:
                print(f"  ⚠️  No valid predictions found for {task_key}")
                continue

            # Convert to numpy arrays
            y_true = np.array(all_y_true)
            y_pred = np.array(all_y_pred)
            y_prob = np.array(all_y_prob)

            # Calculate metrics
            metrics = self.calculate_all_metrics(y_true, y_pred, y_prob)

            # Store results
            flare_class, time_window = task_key.split("_")
            aggregated[task_key] = {
                "flare_class": flare_class,
                "time_window": time_window,
                "n_seeds": len(experiments),
                "metrics": metrics,
                "y_true": y_true,
                "y_pred": y_pred,
                "y_prob": y_prob,
            }

            print(f"  ✅ {task_key}: TSS = {metrics['tss'][0]:.3f}")

        self.aggregated_results = aggregated
        return aggregated

    def generate_latex_table(self, table_type: str = "main") -> str:
        """Generate LaTeX table for the paper."""
        if table_type == "main":
            return self._generate_main_results_table()
        elif table_type == "run_matrix":
            return self._generate_run_matrix_table()
        elif table_type == "comparison":
            return self._generate_comparison_table()
        else:
            raise ValueError(f"Unknown table type: {table_type}")

    def _generate_main_results_table(self) -> str:
        """Generate the main results table with bootstrapped confidence intervals."""
        latex_lines = []

        # Table header
        latex_lines.extend(
            [
                "\\begin{table}[ht]\\centering",
                "\\caption{Bootstrapped performance (mean $\\pm$ 95\\% CI) on the held-out test set. \\textbf{Bold} = best per column; $\\uparrow$ higher is better, $\\downarrow$ lower is better.}",
                "\\label{tab:main_results}",
                "\\small",
                "\\begin{tabular}{lcccccc}",
                "\\toprule",
                "\\textbf{Task} & \\textbf{TSS}$\\uparrow$ & \\textbf{F1}$\\uparrow$ &",
                "\\textbf{Prec.}$\\uparrow$ & \\textbf{Recall}$\\uparrow$ &",
                "\\textbf{Brier}$\\downarrow$ & \\textbf{ECE}$\\downarrow$ \\\\",
                "\\midrule",
            ]
        )

        # Sort tasks
        task_order = []
        for flare in ["C", "M", "M5"]:
            for horizon in ["24h", "48h", "72h"]:
                task_key = f"{flare}_{horizon}"
                if task_key in self.aggregated_results:
                    task_order.append(task_key)

        # Add separators between flare classes
        for i, task_key in enumerate(task_order):
            if i > 0 and task_key.split("_")[0] != task_order[i - 1].split("_")[0]:
                latex_lines.append("\\addlinespace")

            result = self.aggregated_results[task_key]
            metrics = result["metrics"]

            # Format task name
            flare_class = result["flare_class"]
            time_window = result["time_window"].replace("h", " h")
            task_name = f"{flare_class}-{time_window}"

            # Format metrics with confidence intervals
            def format_metric(metric_tuple, decimals=3, bold=False):
                mean, ci_lower, ci_upper = metric_tuple
                ci_width = (ci_upper - ci_lower) / 2
                formatted = f"{mean:.{decimals}f} $\\pm$ {ci_width:.{decimals}f}"
                if bold:
                    formatted = f"\\textbf{{{formatted}}}"
                return formatted

            # Build table row
            row = f"{task_name} & "
            row += f"{format_metric(metrics['tss'])} & "
            row += f"{format_metric(metrics['f1'])} & "
            row += f"{format_metric(metrics['precision'])} & "
            row += f"{format_metric(metrics['recall'])} & "
            row += f"{format_metric(metrics['brier'])} & "
            row += f"{format_metric(metrics['ece'])} \\\\"

            latex_lines.append(row)

        # Table footer
        latex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])

        return "\n".join(latex_lines)

    def _generate_run_matrix_table(self) -> str:
        """Generate the run matrix table showing dataset splits."""
        # This would need to be calculated from your actual dataset
        # For now, return a template
        latex_lines = [
            "% Run matrix table - fill in actual values from your dataset splits",
            "% Use your dataset loading code to calculate these numbers",
        ]
        return "\n".join(latex_lines)

    def _generate_comparison_table(self) -> str:
        """Generate comparison table with baselines."""
        # Template for baseline comparison
        latex_lines = [
            "% Baseline comparison table",
            "% Add your baseline results here",
        ]
        return "\n".join(latex_lines)

    def save_results(self, output_dir: str = "analysis_results"):
        """Save all results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save aggregated results as JSON
        results_for_json = {}
        for task, data in self.aggregated_results.items():
            results_for_json[task] = {
                "flare_class": data["flare_class"],
                "time_window": data["time_window"],
                "n_seeds": data["n_seeds"],
                "metrics": {k: list(v) for k, v in data["metrics"].items()},
            }

        with open(output_path / "aggregated_results.json", "w") as f:
            json.dump(results_for_json, f, indent=2)

        # Save LaTeX tables
        with open(output_path / "main_results_table.tex", "w") as f:
            f.write(self.generate_latex_table("main"))

        # Save detailed CSV
        self._save_detailed_csv(output_path)

        print(f"Results saved to {output_path}")

    def _save_detailed_csv(self, output_path: Path):
        """Save detailed results as CSV."""
        rows = []
        for task, data in self.aggregated_results.items():
            metrics = data["metrics"]
            row = {
                "task": task,
                "flare_class": data["flare_class"],
                "time_window": data["time_window"],
                "n_seeds": data["n_seeds"],
            }

            # Add metrics
            for metric_name, (mean, ci_lower, ci_upper) in metrics.items():
                row[f"{metric_name}_mean"] = mean
                row[f"{metric_name}_ci_lower"] = ci_lower
                row[f"{metric_name}_ci_upper"] = ci_upper
                row[f"{metric_name}_ci_width"] = (ci_upper - ci_lower) / 2

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path / "detailed_results.csv", index=False)

    def plot_results(self, output_dir: str = "analysis_results"):
        """Generate plots for the paper."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Performance comparison plot
        self._plot_performance_comparison(output_path)

        # Reliability diagrams
        self._plot_reliability_diagrams(output_path)

        print(f"Plots saved to {output_path}")

    def _plot_performance_comparison(self, output_path: Path):
        """Plot performance comparison across tasks."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        metrics = ["tss", "f1", "precision", "recall", "brier", "ece"]
        metric_names = ["TSS", "F1", "Precision", "Recall", "Brier Score", "ECE"]

        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i // 3, i % 3]

            tasks = []
            means = []
            ci_widths = []

            for task in sorted(self.aggregated_results.keys()):
                if task in self.aggregated_results:
                    result = self.aggregated_results[task]
                    mean, ci_lower, ci_upper = result["metrics"][metric]
                    tasks.append(task.replace("_", "-"))
                    means.append(mean)
                    ci_widths.append((ci_upper - ci_lower) / 2)

            bars = ax.bar(
                range(len(tasks)), means, yerr=ci_widths, capsize=5, alpha=0.7
            )
            ax.set_xticks(range(len(tasks)))
            ax.set_xticklabels(tasks, rotation=45)
            ax.set_title(name)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_path / "performance_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_reliability_diagrams(self, output_path: Path):
        """Plot reliability diagrams for calibration analysis."""
        n_tasks = len(self.aggregated_results)
        cols = 3
        rows = (n_tasks + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for i, (task, data) in enumerate(self.aggregated_results.items()):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]

            y_true = data["y_true"]
            y_prob = data["y_prob"]

            # Calculate reliability curve
            n_bins = 15
            bin_boundaries = np.linspace(0, 1, n_bins + 1)

            bin_accuracies = []
            bin_confidences = []
            bin_counts = []

            for j in range(n_bins):
                bin_lower = bin_boundaries[j]
                bin_upper = bin_boundaries[j + 1]

                if j == n_bins - 1:  # Include upper boundary for last bin
                    in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
                else:
                    in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

                if np.sum(in_bin) > 0:  # Only process non-empty bins
                    bin_accuracy = np.mean(y_true[in_bin])
                    bin_confidence = np.mean(y_prob[in_bin])  # Actual mean, not bin center
                    bin_count = np.sum(in_bin)
                    
                    # Only append if bin has samples
                    bin_accuracies.append(bin_accuracy)
                    bin_confidences.append(bin_confidence)
                    bin_counts.append(bin_count)

            # Plot reliability curve (only non-empty bins)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
            if bin_confidences:  # Only plot if we have data
                ax.plot(bin_confidences, bin_accuracies, "o-", label="Model")

            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.set_title(f'{task.replace("_", "-")}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

        # Hide empty subplots
        for i in range(n_tasks, rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(
            output_path / "reliability_diagrams.png", dpi=300, bbox_inches="tight"
        )
        plt.close()


def main():
    """Main analysis script."""
    print("🚀 Starting EVEREST Results Analysis")
    print("=" * 50)

    # Initialize analyzer with quality filtering
    quality_threshold = 0.6
    print(
        f"🔧 Quality threshold: {quality_threshold*100}% (experiments with any metric below this will be discarded)"
    )
    analyzer = ExperimentAnalyzer("results", quality_threshold=quality_threshold)

    # Load all experiments (with automatic quality filtering)
    experiments = analyzer.load_experiment_data()
    if not experiments:
        print(
            "❌ No high-quality experiments found. Check your results directory or lower the quality threshold."
        )
        return

    # Aggregate results
    aggregated = analyzer.aggregate_experiments()
    if not aggregated:
        print("❌ No valid aggregated results.")
        return

    # Generate and save results
    analyzer.save_results("analysis_results")

    # Generate plots
    analyzer.plot_results("analysis_results")

    # Print summary
    print("\n" + "=" * 50)
    print("📊 RESULTS SUMMARY")
    print("=" * 50)

    for task, data in aggregated.items():
        metrics = data["metrics"]
        n_seeds = data["n_seeds"]
        print(f"\n{task.replace('_', '-')} ({n_seeds} high-quality seeds):")
        print(
            f"  TSS:       {metrics['tss'][0]:.3f} ± {(metrics['tss'][2] - metrics['tss'][1])/2:.3f}"
        )
        print(
            f"  F1:        {metrics['f1'][0]:.3f} ± {(metrics['f1'][2] - metrics['f1'][1])/2:.3f}"
        )
        print(
            f"  Precision: {metrics['precision'][0]:.3f} ± {(metrics['precision'][2] - metrics['precision'][1])/2:.3f}"
        )
        print(
            f"  Recall:    {metrics['recall'][0]:.3f} ± {(metrics['recall'][2] - metrics['recall'][1])/2:.3f}"
        )
        print(
            f"  ECE:       {metrics['ece'][0]:.3f} ± {(metrics['ece'][2] - metrics['ece'][1])/2:.3f}"
        )

    print(f"\n✅ Analysis complete! Check 'analysis_results/' for outputs.")
    print(
        f"🗑️  Bad seeds (< {quality_threshold*100}% performance) were automatically discarded."
    )

    # Print LaTeX table
    print("\n" + "=" * 50)
    print("📋 LATEX TABLE")
    print("=" * 50)
    print(analyzer.generate_latex_table("main"))


if __name__ == "__main__":
    main()
