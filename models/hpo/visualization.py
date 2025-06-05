"""
Visualization utilities for EVEREST Hyperparameter Optimization results.

This module provides functions to create plots and visualizations for
analyzing HPO study results, parameter importance, and optimization progress.
"""

from .config import OUTPUT_DIRS, SEARCH_STAGES
import os
import sys
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)

# Add models directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class HPOVisualizer:
    """Visualization utilities for HPO study results."""

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory to save plots (defaults to config)
        """
        self.output_dir = Path(output_dir or OUTPUT_DIRS["plots"])
        self.output_dir.mkdir(exist_ok=True)

        # Set up plotting style
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette("husl")

    def plot_study_optimization_history(
        self, study: optuna.Study, flare_class: str, time_window: str, save: bool = True
    ) -> plt.Figure:
        """
        Plot optimization history for a study.

        Args:
            study: Optuna study object
            flare_class: Target flare class
            time_window: Prediction window
            save: Whether to save the plot

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Get completed trials
        trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not trials:
            fig.suptitle(f"No completed trials for {flare_class}-{time_window}h")
            return fig

        trial_numbers = [t.number for t in trials]
        values = [t.value for t in trials]

        # Plot 1: Optimization history
        ax1.plot(trial_numbers, values, "o-", alpha=0.7)
        ax1.axhline(
            y=max(values),
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Best TSS: {max(values):.4f}",
        )
        ax1.set_xlabel("Trial Number")
        ax1.set_ylabel("TSS Score")
        ax1.set_title(
            f"Optimization History: {flare_class}-class, {time_window}h window"
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Best value progression
        best_values = []
        current_best = -float("inf")
        for value in values:
            current_best = max(current_best, value)
            best_values.append(current_best)

        ax2.plot(trial_numbers, best_values, "g-", linewidth=2, label="Best TSS So Far")
        ax2.fill_between(trial_numbers, best_values, alpha=0.3)
        ax2.set_xlabel("Trial Number")
        ax2.set_ylabel("Best TSS Score")
        ax2.set_title("Best Value Progression")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = f"optimization_history_{flare_class}_{time_window}h.png"
            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Optimization history saved to {filepath}")

        return fig

    def plot_parameter_importance(
        self, study: optuna.Study, flare_class: str, time_window: str, save: bool = True
    ) -> plt.Figure:
        """
        Plot parameter importance for a study.

        Args:
            study: Optuna study object
            flare_class: Target flare class
            time_window: Prediction window
            save: Whether to save the plot

        Returns:
            Matplotlib figure
        """
        # Get parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
        except BaseException:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "Not enough completed trials for importance analysis",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"Parameter Importance: {flare_class}-{time_window}h")
            return fig

        if not importance:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No parameter importance data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"Parameter Importance: {flare_class}-{time_window}h")
            return fig

        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))

        params = list(importance.keys())
        values = list(importance.values())

        bars = ax.barh(params, values, color="skyblue", alpha=0.8)
        ax.set_xlabel("Importance")
        ax.set_title(
            f"Parameter Importance: {flare_class}-class, {time_window}h window"
        )
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                ha="left",
                va="center",
            )

        plt.tight_layout()

        if save:
            filename = f"parameter_importance_{flare_class}_{time_window}h.png"
            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Parameter importance saved to {filepath}")

        return fig

    def plot_parallel_coordinates(
        self,
        study: optuna.Study,
        flare_class: str,
        time_window: str,
        n_trials: int = 50,
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot parallel coordinates for top trials.

        Args:
            study: Optuna study object
            flare_class: Target flare class
            time_window: Prediction window
            n_trials: Number of top trials to include
            save: Whether to save the plot

        Returns:
            Matplotlib figure
        """
        # Get top trials
        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if len(completed_trials) < 5:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(
                0.5,
                0.5,
                "Not enough completed trials for parallel coordinates",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"Parallel Coordinates: {flare_class}-{time_window}h")
            return fig

        top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[
            :n_trials
        ]

        # Extract data
        param_names = list(top_trials[0].params.keys())
        data = []

        for trial in top_trials:
            row = [trial.params[param] for param in param_names] + [trial.value]
            data.append(row)

        df = pd.DataFrame(data, columns=param_names + ["TSS"])

        # Create parallel coordinates plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Normalize data for plotting
        df_norm = df.copy()
        for col in df_norm.columns:
            if col != "TSS":
                df_norm[col] = (df_norm[col] - df_norm[col].min()) / (
                    df_norm[col].max() - df_norm[col].min()
                )

        # Plot lines
        for i, (_, row) in enumerate(df_norm.iterrows()):
            color = plt.cm.viridis(row["TSS"] / df["TSS"].max())
            ax.plot(
                range(len(param_names)), row[param_names], "o-", alpha=0.7, color=color
            )

        ax.set_xticks(range(len(param_names)))
        ax.set_xticklabels(param_names, rotation=45)
        ax.set_ylabel("Normalized Parameter Value")
        ax.set_title(
            f"Parallel Coordinates: Top {n_trials} trials for {flare_class}-{time_window}h"
        )
        ax.grid(True, alpha=0.3)

        # Add colorbar
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis,
            norm=plt.Normalize(vmin=df["TSS"].min(), vmax=df["TSS"].max()),
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("TSS Score")

        plt.tight_layout()

        if save:
            filename = f"parallel_coordinates_{flare_class}_{time_window}h.png"
            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Parallel coordinates saved to {filepath}")

        return fig

    def plot_stage_analysis(
        self, study: optuna.Study, flare_class: str, time_window: str, save: bool = True
    ) -> plt.Figure:
        """
        Plot analysis by optimization stage.

        Args:
            study: Optuna study object
            flare_class: Target flare class
            time_window: Prediction window
            save: Whether to save the plot

        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Collect stage data
        stage_data = {
            stage: {"trials": [], "values": []} for stage in SEARCH_STAGES.keys()
        }
        stage_data["unknown"] = {"trials": [], "values": []}

        for trial in study.trials:
            stage = trial.user_attrs.get("stage", "unknown")
            stage_data[stage]["trials"].append(trial.number)
            if trial.state == optuna.trial.TrialState.COMPLETE:
                stage_data[stage]["values"].append(trial.value)
            else:
                stage_data[stage]["values"].append(None)

        # Plot 1: Trials by stage
        stage_counts = {
            stage: len(data["trials"])
            for stage, data in stage_data.items()
            if data["trials"]
        }
        if stage_counts:
            ax1.bar(stage_counts.keys(), stage_counts.values(), alpha=0.7)
            ax1.set_title("Number of Trials by Stage")
            ax1.set_ylabel("Number of Trials")

        # Plot 2: Best values by stage
        stage_best = {}
        for stage, data in stage_data.items():
            valid_values = [v for v in data["values"] if v is not None]
            if valid_values:
                stage_best[stage] = max(valid_values)

        if stage_best:
            ax2.bar(stage_best.keys(), stage_best.values(), alpha=0.7, color="green")
            ax2.set_title("Best TSS by Stage")
            ax2.set_ylabel("Best TSS Score")

        # Plot 3: Value distribution by stage
        for stage, data in stage_data.items():
            valid_values = [v for v in data["values"] if v is not None]
            if len(valid_values) > 1:
                ax3.hist(valid_values, alpha=0.6, label=stage, bins=20)

        ax3.set_title("TSS Distribution by Stage")
        ax3.set_xlabel("TSS Score")
        ax3.set_ylabel("Frequency")
        ax3.legend()

        # Plot 4: Progress over time
        for stage, data in stage_data.items():
            if len(data["trials"]) > 0:
                x = data["trials"]
                y = [v if v is not None else 0 for v in data["values"]]
                ax4.scatter(x, y, alpha=0.6, label=stage, s=30)

        ax4.set_title("Trial Progress by Stage")
        ax4.set_xlabel("Trial Number")
        ax4.set_ylabel("TSS Score")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        fig.suptitle(
            f"Stage Analysis: {flare_class}-class, {time_window}h window", fontsize=16
        )
        plt.tight_layout()

        if save:
            filename = f"stage_analysis_{flare_class}_{time_window}h.png"
            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Stage analysis saved to {filepath}")

        return fig

    def plot_combined_summary(
        self, results: Dict[str, Dict[str, Any]], save: bool = True
    ) -> plt.Figure:
        """
        Plot combined summary across all targets.

        Args:
            results: Combined results from all studies
            save: Whether to save the plot

        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Extract data
        targets = []
        best_tss = []
        n_trials = []
        optimization_times = []

        for target_key, result in results.items():
            if "best_trial" in result:
                targets.append(target_key)
                best_tss.append(result["best_trial"]["value"])
                n_trials.append(result["n_trials"])
                optimization_times.append(result["optimization_time"])

        if not targets:
            fig.suptitle("No valid results to display")
            return fig

        # Plot 1: Best TSS by target
        y_pos = np.arange(len(targets))
        bars1 = ax1.barh(y_pos, best_tss, alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(targets)
        ax1.set_xlabel("Best TSS Score")
        ax1.set_title("Best TSS Score by Target")
        ax1.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for bar, value in zip(bars1, best_tss):
            ax1.text(
                bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                ha="left",
                va="center",
            )

        # Plot 2: Number of trials by target
        bars2 = ax2.bar(range(len(targets)), n_trials, alpha=0.8, color="orange")
        ax2.set_xticks(range(len(targets)))
        ax2.set_xticklabels(targets, rotation=45)
        ax2.set_ylabel("Number of Trials")
        ax2.set_title("Number of Trials by Target")
        ax2.grid(True, alpha=0.3, axis="y")

        # Plot 3: Optimization time by target
        bars3 = ax3.bar(
            range(len(targets)),
            [t / 3600 for t in optimization_times],
            alpha=0.8,
            color="green",
        )  # Convert to hours
        ax3.set_xticks(range(len(targets)))
        ax3.set_xticklabels(targets, rotation=45)
        ax3.set_ylabel("Optimization Time (hours)")
        ax3.set_title("Optimization Time by Target")
        ax3.grid(True, alpha=0.3, axis="y")

        # Plot 4: TSS vs optimization time
        scatter = ax4.scatter(
            [t / 3600 for t in optimization_times],
            best_tss,
            s=100,
            alpha=0.7,
            c=range(len(targets)),
            cmap="viridis",
        )
        ax4.set_xlabel("Optimization Time (hours)")
        ax4.set_ylabel("Best TSS Score")
        ax4.set_title("TSS Score vs Optimization Time")
        ax4.grid(True, alpha=0.3)

        # Add target labels to scatter plot
        for i, target in enumerate(targets):
            ax4.annotate(
                target,
                (optimization_times[i] / 3600, best_tss[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
            )

        fig.suptitle("HPO Summary Across All Targets", fontsize=16)
        plt.tight_layout()

        if save:
            filename = "hpo_combined_summary.png"
            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Combined summary saved to {filepath}")

        return fig

    def create_all_plots(
        self, study: optuna.Study, flare_class: str, time_window: str
    ) -> None:
        """Create all plots for a study."""
        print(f"Creating visualizations for {flare_class}-{time_window}h...")

        # Create individual plots
        self.plot_study_optimization_history(study, flare_class, time_window)
        self.plot_parameter_importance(study, flare_class, time_window)
        self.plot_parallel_coordinates(study, flare_class, time_window)
        self.plot_stage_analysis(study, flare_class, time_window)

        print(f"All visualizations saved to {self.output_dir}")


def load_study_from_file(study_file: str) -> optuna.Study:
    """Load a study from pickle file."""
    import pickle

    with open(study_file, "rb") as f:
        return pickle.load(f)


def main():
    """Main function for creating visualizations."""
    import argparse

    parser = argparse.ArgumentParser(description="Create HPO visualizations")
    parser.add_argument("--study-file", type=str, help="Path to study pickle file")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="models/hpo/results",
        help="Results directory to scan",
    )
    parser.add_argument("--flare-class", type=str, help="Specific flare class")
    parser.add_argument("--time-window", type=str, help="Specific time window")

    args = parser.parse_args()

    visualizer = HPOVisualizer()

    if args.study_file:
        # Single study visualization
        study = load_study_from_file(args.study_file)
        flare_class = args.flare_class or "M"
        time_window = args.time_window or "24"
        visualizer.create_all_plots(study, flare_class, time_window)

    else:
        # Scan results directory
        results_dir = Path(args.results_dir)
        if results_dir.exists():
            for target_dir in results_dir.iterdir():
                if target_dir.is_dir():
                    study_file = target_dir / "study.pkl"
                    if study_file.exists():
                        parts = target_dir.name.split("_")
                        if len(parts) >= 2:
                            flare_class = parts[0]
                            time_window = parts[1].replace("h", "")
                            study = load_study_from_file(str(study_file))
                            visualizer.create_all_plots(study, flare_class, time_window)


if __name__ == "__main__":
    main()
