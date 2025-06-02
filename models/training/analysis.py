"""
EVEREST Production Training Analysis

This module provides comprehensive statistical analysis and visualization
of production training results across all flare class × time window combinations.
"""

from .config import (
    TRAINING_TARGETS, RANDOM_SEEDS, EVALUATION_METRICS, OUTPUT_CONFIG,
    STATISTICAL_CONFIG, get_all_experiments
)
import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
from scipy import stats
from sklearn.metrics import confusion_matrix

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class ProductionAnalyzer:
    """Comprehensive analyzer for production training results."""

    def __init__(self):
        """Initialize analyzer."""
        self.results_dir = OUTPUT_CONFIG["results_dir"]
        self.plots_dir = OUTPUT_CONFIG["plots_dir"]
        self.analysis_dir = OUTPUT_CONFIG["analysis_dir"]

        # Create output directories
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)

        print("📊 Production Training Analyzer initialized")
        print(f"   Results directory: {self.results_dir}")
        print(f"   Plots directory: {self.plots_dir}")
        print(f"   Analysis directory: {self.analysis_dir}")

    def load_all_results(self) -> pd.DataFrame:
        """Load all experiment results into a DataFrame."""
        print("\n📂 Loading experiment results...")

        all_results = []
        experiments = get_all_experiments()

        for experiment in experiments:
            experiment_dir = os.path.join(self.results_dir, experiment["experiment_name"])
            results_file = os.path.join(experiment_dir, "results.json")

            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        data = json.load(f)

                    # Extract key metrics
                    result_row = {
                        "experiment_name": experiment["experiment_name"],
                        "flare_class": experiment["flare_class"],
                        "time_window": int(experiment["time_window"]),
                        "seed": experiment["seed"],
                        "optimal_threshold": data["threshold_optimization"]["optimal_threshold"],
                        **data["evaluation"]["test_metrics"]
                    }

                    # Add training time if available
                    if "training" in data and "training_time" in data["training"]:
                        result_row["training_time"] = data["training"]["training_time"]

                    all_results.append(result_row)

                except Exception as e:
                    print(f"⚠️  Failed to load {experiment['experiment_name']}: {e}")
            else:
                print(f"⚠️  Missing results for {experiment['experiment_name']}")

        df = pd.DataFrame(all_results)

        print(f"✅ Loaded {len(df)} experiment results")
        print(f"   Targets: {df['flare_class'].nunique()} flare classes × {df['time_window'].nunique()} time windows")
        print(f"   Seeds per target: {df.groupby(['flare_class', 'time_window']).size().iloc[0] if len(df) > 0 else 0}")

        return df

    def calculate_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate summary statistics across seeds for each target."""
        print("\n📈 Calculating summary statistics...")

        # Group by target (flare_class, time_window)
        grouped = df.groupby(['flare_class', 'time_window'])

        summary_stats = []

        for (flare_class, time_window), group in grouped:
            target_stats = {
                "flare_class": flare_class,
                "time_window": time_window,
                "n_seeds": len(group)
            }

            # Calculate statistics for each metric
            for metric in EVALUATION_METRICS:
                if metric in group.columns:
                    values = group[metric].values
                    target_stats.update({
                        f"{metric}_mean": np.mean(values),
                        f"{metric}_std": np.std(values, ddof=1),
                        f"{metric}_min": np.min(values),
                        f"{metric}_max": np.max(values),
                        f"{metric}_median": np.median(values)
                    })

                    # Calculate 95% confidence interval
                    if len(values) > 1:
                        ci = stats.t.interval(
                            STATISTICAL_CONFIG["confidence_level"],
                            len(values) - 1,
                            loc=np.mean(values),
                            scale=stats.sem(values)
                        )
                        target_stats.update({
                            f"{metric}_ci_lower": ci[0],
                            f"{metric}_ci_upper": ci[1]
                        })

            # Calculate optimal threshold statistics
            if "optimal_threshold" in group.columns:
                thresholds = group["optimal_threshold"].values
                target_stats.update({
                    "threshold_mean": np.mean(thresholds),
                    "threshold_std": np.std(thresholds, ddof=1),
                    "threshold_min": np.min(thresholds),
                    "threshold_max": np.max(thresholds)
                })

            summary_stats.append(target_stats)

        summary_df = pd.DataFrame(summary_stats)

        print(f"✅ Summary statistics calculated for {len(summary_df)} targets")

        return summary_df

    def create_performance_comparison(self, summary_df: pd.DataFrame):
        """Create comprehensive performance comparison visualizations."""
        print("\n📊 Creating performance comparison plots...")

        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. TSS comparison across all targets
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('EVEREST Production Model Performance Analysis', fontsize=16, fontweight='bold')

        # TSS by flare class and time window
        ax1 = axes[0, 0]
        pivot_tss = summary_df.pivot(index='flare_class', columns='time_window', values='tss_mean')
        pivot_tss_std = summary_df.pivot(index='flare_class', columns='time_window', values='tss_std')

        im1 = ax1.imshow(pivot_tss.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax1.set_xticks(range(len(pivot_tss.columns)))
        ax1.set_yticks(range(len(pivot_tss.index)))
        ax1.set_xticklabels([f'{col}h' for col in pivot_tss.columns])
        ax1.set_yticklabels(pivot_tss.index)
        ax1.set_title('True Skill Statistic (TSS)')
        ax1.set_xlabel('Time Window')
        ax1.set_ylabel('Flare Class')

        # Add text annotations
        for i in range(len(pivot_tss.index)):
            for j in range(len(pivot_tss.columns)):
                mean_val = pivot_tss.iloc[i, j]
                std_val = pivot_tss_std.iloc[i, j]
                if not np.isnan(mean_val):
                    ax1.text(j, i, f'{mean_val:.3f}\n±{std_val:.3f}',
                             ha='center', va='center', fontsize=9, fontweight='bold')

        plt.colorbar(im1, ax=ax1, label='TSS')

        # F1 Score comparison
        ax2 = axes[0, 1]
        pivot_f1 = summary_df.pivot(index='flare_class', columns='time_window', values='f1_mean')
        pivot_f1_std = summary_df.pivot(index='flare_class', columns='time_window', values='f1_std')

        im2 = ax2.imshow(pivot_f1.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax2.set_xticks(range(len(pivot_f1.columns)))
        ax2.set_yticks(range(len(pivot_f1.index)))
        ax2.set_xticklabels([f'{col}h' for col in pivot_f1.columns])
        ax2.set_yticklabels(pivot_f1.index)
        ax2.set_title('F1 Score')
        ax2.set_xlabel('Time Window')
        ax2.set_ylabel('Flare Class')

        for i in range(len(pivot_f1.index)):
            for j in range(len(pivot_f1.columns)):
                mean_val = pivot_f1.iloc[i, j]
                std_val = pivot_f1_std.iloc[i, j]
                if not np.isnan(mean_val):
                    ax2.text(j, i, f'{mean_val:.3f}\n±{std_val:.3f}',
                             ha='center', va='center', fontsize=9, fontweight='bold')

        plt.colorbar(im2, ax=ax2, label='F1 Score')

        # Threshold distribution
        ax3 = axes[1, 0]
        for flare_class in summary_df['flare_class'].unique():
            class_data = summary_df[summary_df['flare_class'] == flare_class]
            ax3.plot(class_data['time_window'], class_data['threshold_mean'],
                     marker='o', label=f'{flare_class}-class', linewidth=2, markersize=8)
            ax3.fill_between(class_data['time_window'],
                             class_data['threshold_mean'] - class_data['threshold_std'],
                             class_data['threshold_mean'] + class_data['threshold_std'],
                             alpha=0.2)

        ax3.set_xlabel('Time Window (hours)')
        ax3.set_ylabel('Optimal Threshold')
        ax3.set_title('Optimal Classification Thresholds')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Performance vs Latency
        ax4 = axes[1, 1]
        scatter = ax4.scatter(summary_df['latency_ms_mean'], summary_df['tss_mean'],
                              c=summary_df['time_window'], s=100, alpha=0.7, cmap='viridis')

        # Add labels for each point
        for _, row in summary_df.iterrows():
            ax4.annotate(f"{row['flare_class']}-{row['time_window']}h",
                         (row['latency_ms_mean'], row['tss_mean']),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax4.set_xlabel('Inference Latency (ms)')
        ax4.set_ylabel('TSS')
        ax4.set_title('Performance vs Latency Trade-off')
        plt.colorbar(scatter, ax=ax4, label='Time Window (h)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot
        plot_file = os.path.join(self.plots_dir, "production_performance_analysis.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Performance comparison saved: {plot_file}")

    def create_detailed_metrics_plots(self, df: pd.DataFrame):
        """Create detailed plots for all metrics."""
        print("\n📊 Creating detailed metrics plots...")

        # Create box plots for each metric
        metrics_to_plot = ['tss', 'f1', 'precision', 'recall', 'roc_auc', 'brier', 'ece']

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('EVEREST Production Model - Detailed Metrics Analysis', fontsize=16, fontweight='bold')

        for i, metric in enumerate(metrics_to_plot):
            if i < len(axes.flat) and metric in df.columns:
                ax = axes.flat[i]

                # Create box plot grouped by flare class and time window
                df['target'] = df['flare_class'] + '-' + df['time_window'].astype(str) + 'h'

                box_data = []
                labels = []
                for target in sorted(df['target'].unique()):
                    target_data = df[df['target'] == target][metric].values
                    if len(target_data) > 0:
                        box_data.append(target_data)
                        labels.append(target)

                bp = ax.boxplot(box_data, labels=labels, patch_artist=True)

                # Color boxes by flare class
                colors = {'C': 'lightblue', 'M': 'lightgreen', 'M5': 'lightcoral'}
                for patch, label in zip(bp['boxes'], labels):
                    flare_class = label.split('-')[0]
                    patch.set_facecolor(colors.get(flare_class, 'lightgray'))

                ax.set_title(f'{metric.upper()}')
                ax.set_ylabel(metric.upper())
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for i in range(len(metrics_to_plot), len(axes.flat)):
            axes.flat[i].remove()

        plt.tight_layout()

        # Save the plot
        plot_file = os.path.join(self.plots_dir, "detailed_metrics_analysis.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Detailed metrics plot saved: {plot_file}")

    def create_threshold_analysis(self, df: pd.DataFrame):
        """Create threshold optimization analysis."""
        print("\n🎯 Creating threshold analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Classification Threshold Analysis', fontsize=16, fontweight='bold')

        # Threshold distribution by target
        ax1 = axes[0, 0]
        df['target'] = df['flare_class'] + '-' + df['time_window'].astype(str) + 'h'

        for target in sorted(df['target'].unique()):
            target_data = df[df['target'] == target]['optimal_threshold']
            ax1.hist(target_data, alpha=0.6, label=target, bins=10)

        ax1.set_xlabel('Optimal Threshold')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Threshold Distribution by Target')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Threshold vs TSS correlation
        ax2 = axes[0, 1]
        scatter = ax2.scatter(df['optimal_threshold'], df['tss'],
                              c=df['time_window'], s=60, alpha=0.7, cmap='viridis')
        ax2.set_xlabel('Optimal Threshold')
        ax2.set_ylabel('TSS')
        ax2.set_title('Threshold vs TSS Performance')
        plt.colorbar(scatter, ax=ax2, label='Time Window (h)')
        ax2.grid(True, alpha=0.3)

        # Threshold stability across seeds
        ax3 = axes[1, 0]
        summary_df = df.groupby(['flare_class', 'time_window']).agg({
            'optimal_threshold': ['mean', 'std']
        }).round(4)
        summary_df.columns = ['threshold_mean', 'threshold_std']
        summary_df = summary_df.reset_index()

        x_pos = np.arange(len(summary_df))
        bars = ax3.bar(x_pos, summary_df['threshold_mean'],
                       yerr=summary_df['threshold_std'], capsize=5)

        # Color bars by flare class
        colors = {'C': 'lightblue', 'M': 'lightgreen', 'M5': 'lightcoral'}
        for bar, (_, row) in zip(bars, summary_df.iterrows()):
            bar.set_color(colors.get(row['flare_class'], 'lightgray'))

        ax3.set_xlabel('Target')
        ax3.set_ylabel('Optimal Threshold')
        ax3.set_title('Threshold Stability Across Seeds')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f"{row['flare_class']}-{row['time_window']}h"
                             for _, row in summary_df.iterrows()], rotation=45)
        ax3.grid(True, alpha=0.3)

        # Threshold impact on different metrics
        ax4 = axes[1, 1]
        metrics = ['precision', 'recall', 'f1']

        for metric in metrics:
            if metric in df.columns:
                correlation = df['optimal_threshold'].corr(df[metric])
                ax4.scatter(df['optimal_threshold'], df[metric],
                            alpha=0.6, label=f'{metric} (r={correlation:.3f})')

        ax4.set_xlabel('Optimal Threshold')
        ax4.set_ylabel('Metric Value')
        ax4.set_title('Threshold Impact on Key Metrics')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot
        plot_file = os.path.join(self.plots_dir, "threshold_analysis.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Threshold analysis saved: {plot_file}")

    def generate_summary_report(self, df: pd.DataFrame, summary_df: pd.DataFrame):
        """Generate comprehensive summary report."""
        print("\n📝 Generating summary report...")

        report_lines = []
        report_lines.append("# EVEREST Production Training Summary Report")
        report_lines.append("=" * 60)
        report_lines.append("")

        # Overall statistics
        report_lines.append("## Overall Statistics")
        report_lines.append(f"- Total experiments: {len(df)}")
        report_lines.append(f"- Targets: {len(summary_df)} (flare class × time window combinations)")
        report_lines.append(f"- Seeds per target: {df.groupby(['flare_class', 'time_window']).size().iloc[0] if len(df) > 0 else 0}")
        report_lines.append("")

        # Best performing models
        report_lines.append("## Best Performing Models")
        best_tss = summary_df.loc[summary_df['tss_mean'].idxmax()]
        best_f1 = summary_df.loc[summary_df['f1_mean'].idxmax()]

        report_lines.append(f"### Highest TSS: {best_tss['flare_class']}-{best_tss['time_window']}h")
        report_lines.append(f"- TSS: {best_tss['tss_mean']:.4f} ± {best_tss['tss_std']:.4f}")
        report_lines.append(f"- F1: {best_tss['f1_mean']:.4f} ± {best_tss['f1_std']:.4f}")
        report_lines.append(f"- Threshold: {best_tss['threshold_mean']:.3f} ± {best_tss['threshold_std']:.3f}")
        report_lines.append("")

        report_lines.append(f"### Highest F1: {best_f1['flare_class']}-{best_f1['time_window']}h")
        report_lines.append(f"- F1: {best_f1['f1_mean']:.4f} ± {best_f1['f1_std']:.4f}")
        report_lines.append(f"- TSS: {best_f1['tss_mean']:.4f} ± {best_f1['tss_std']:.4f}")
        report_lines.append(f"- Threshold: {best_f1['threshold_mean']:.3f} ± {best_f1['threshold_std']:.3f}")
        report_lines.append("")

        # Performance by flare class
        report_lines.append("## Performance by Flare Class")
        for flare_class in sorted(summary_df['flare_class'].unique()):
            class_data = summary_df[summary_df['flare_class'] == flare_class]
            avg_tss = class_data['tss_mean'].mean()
            avg_f1 = class_data['f1_mean'].mean()

            report_lines.append(f"### {flare_class}-class")
            report_lines.append(f"- Average TSS: {avg_tss:.4f}")
            report_lines.append(f"- Average F1: {avg_f1:.4f}")
            report_lines.append(f"- Time windows: {sorted(class_data['time_window'].tolist())}")
            report_lines.append("")

        # Performance by time window
        report_lines.append("## Performance by Time Window")
        for time_window in sorted(summary_df['time_window'].unique()):
            window_data = summary_df[summary_df['time_window'] == time_window]
            avg_tss = window_data['tss_mean'].mean()
            avg_f1 = window_data['f1_mean'].mean()

            report_lines.append(f"### {time_window}-hour window")
            report_lines.append(f"- Average TSS: {avg_tss:.4f}")
            report_lines.append(f"- Average F1: {avg_f1:.4f}")
            report_lines.append(f"- Flare classes: {sorted(window_data['flare_class'].tolist())}")
            report_lines.append("")

        # Detailed results table
        report_lines.append("## Detailed Results")
        report_lines.append("| Target | TSS | F1 | Precision | Recall | ROC AUC | Threshold | Latency (ms) |")
        report_lines.append("|--------|-----|----|-----------|---------|---------|-----------|--------------| ")

        for _, row in summary_df.iterrows():
            target = f"{row['flare_class']}-{row['time_window']}h"
            tss = f"{row['tss_mean']:.3f}±{row['tss_std']:.3f}"
            f1 = f"{row['f1_mean']:.3f}±{row['f1_std']:.3f}"
            precision = f"{row['precision_mean']:.3f}±{row['precision_std']:.3f}"
            recall = f"{row['recall_mean']:.3f}±{row['recall_std']:.3f}"
            roc_auc = f"{row['roc_auc_mean']:.3f}±{row['roc_auc_std']:.3f}"
            threshold = f"{row['threshold_mean']:.3f}±{row['threshold_std']:.3f}"
            latency = f"{row['latency_ms_mean']:.1f}±{row['latency_ms_std']:.1f}"

            report_lines.append(f"| {target} | {tss} | {f1} | {precision} | {recall} | {roc_auc} | {threshold} | {latency} |")

        report_lines.append("")

        # Save report
        report_file = os.path.join(self.analysis_dir, "production_training_report.md")
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"✅ Summary report saved: {report_file}")

    def save_results_csv(self, df: pd.DataFrame, summary_df: pd.DataFrame):
        """Save results as CSV files."""
        print("\n💾 Saving CSV files...")

        # Save raw results
        raw_file = os.path.join(self.analysis_dir, "raw_results.csv")
        df.to_csv(raw_file, index=False)

        # Save summary statistics
        summary_file = os.path.join(self.analysis_dir, "summary_statistics.csv")
        summary_df.to_csv(summary_file, index=False)

        print(f"✅ Raw results saved: {raw_file}")
        print(f"✅ Summary statistics saved: {summary_file}")

    def run_complete_analysis(self):
        """Run complete analysis pipeline."""
        print("🚀 Starting complete production training analysis")
        print("=" * 60)

        # Load results
        df = self.load_all_results()

        if len(df) == 0:
            print("❌ No results found. Please ensure training experiments have completed.")
            return

        # Calculate summary statistics
        summary_df = self.calculate_summary_statistics(df)

        # Create visualizations
        self.create_performance_comparison(summary_df)
        self.create_detailed_metrics_plots(df)
        self.create_threshold_analysis(df)

        # Generate reports
        self.generate_summary_report(df, summary_df)
        self.save_results_csv(df, summary_df)

        print("\n🎉 Complete analysis finished!")
        print(f"📁 Results saved to: {self.analysis_dir}")
        print(f"📊 Plots saved to: {self.plots_dir}")


def main():
    """Main entry point."""
    analyzer = ProductionAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
