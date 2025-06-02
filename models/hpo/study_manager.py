"""
Study Manager for EVEREST Hyperparameter Optimization

This module orchestrates the three-stage Bayesian search using Optuna v3.6
with Ray Tune for asynchronous execution and median-stopping pruner.
"""

from .objective import create_objective
from .config import (
    OPTUNA_CONFIG, RAY_TUNE_CONFIG, SEARCH_STAGES, OUTPUT_DIRS,
    REPRODUCIBILITY_CONFIG, EXPERIMENT_TARGETS, get_total_trials,
    create_output_dirs
)
import os
import sys
import json
import time
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import optuna
import numpy as np
import pandas as pd
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Add models directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class StudyManager:
    """
    Manages EVEREST hyperparameter optimization studies with three-stage protocol.

    This class handles:
    - Study creation and configuration
    - Three-stage optimization protocol
    - Results tracking and analysis
    - Reproducibility and logging
    """

    def __init__(self, study_name: Optional[str] = None, storage_url: Optional[str] = None):
        """
        Initialize the study manager.

        Args:
            study_name: Name for the Optuna study (defaults to config)
            storage_url: Database URL for study persistence (defaults to config)
        """
        self.study_name = study_name or OPTUNA_CONFIG["study_name"]
        self.storage_url = storage_url or OPTUNA_CONFIG["storage"]

        # Set up output directories
        create_output_dirs()

        # Set up logging
        self._setup_logging()

        # Set up reproducibility
        self._setup_reproducibility()

        # Initialize study tracking
        self.studies = {}  # flare_class_time_window -> study
        self.results = {}  # flare_class_time_window -> results

    def _setup_logging(self) -> None:
        """Configure logging for the study."""
        log_dir = Path(OUTPUT_DIRS["logs"])
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"hpo_study_{timestamp}.log"

        logging.basicConfig(
            level=getattr(logging, REPRODUCIBILITY_CONFIG["log_level"]),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger("StudyManager")
        self.logger.info(f"Study manager initialized. Logs: {log_file}")

    def _setup_reproducibility(self) -> None:
        """Set up reproducible environment."""
        seed = REPRODUCIBILITY_CONFIG["random_seed"]

        # Set random seeds
        np.random.seed(seed)
        optuna.logging.set_verbosity(optuna.logging.INFO)

        self.logger.info(f"Reproducibility configured with seed: {seed}")

    def create_study(self, flare_class: str, time_window: str) -> optuna.Study:
        """
        Create an Optuna study for specific target configuration.

        Args:
            flare_class: Target flare class
            time_window: Prediction window

        Returns:
            Configured Optuna study
        """
        study_name = f"{self.study_name}_{flare_class}_{time_window}h"

        # Configure sampler with reproducible seed
        sampler = TPESampler(
            seed=REPRODUCIBILITY_CONFIG["random_seed"],
            n_startup_trials=10,  # Random sampling for first 10 trials
            n_ei_candidates=24,   # Number of candidates for EI acquisition
            multivariate=True,    # Use multivariate TPE
            warn_independent_sampling=False  # Suppress warnings for large studies
        )

        # Configure pruner
        pruner = MedianPruner(
            n_startup_trials=OPTUNA_CONFIG["pruner_config"]["n_startup_trials"],
            n_warmup_steps=OPTUNA_CONFIG["pruner_config"]["n_warmup_steps"],
            interval_steps=OPTUNA_CONFIG["pruner_config"]["interval_steps"]
        )

        try:
            # Try to load existing study
            study = optuna.load_study(
                study_name=study_name,
                storage=self.storage_url,
                sampler=sampler,
                pruner=pruner
            )
            self.logger.info(f"Loaded existing study: {study_name}")

        except KeyError:
            # Create new study
            study = optuna.create_study(
                study_name=study_name,
                direction=OPTUNA_CONFIG["direction"],
                storage=self.storage_url,
                sampler=sampler,
                pruner=pruner
            )
            self.logger.info(f"Created new study: {study_name}")

        return study

    def run_single_target(
        self,
        flare_class: str,
        time_window: str,
        max_trials: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run optimization for a single target configuration.

        Args:
            flare_class: Target flare class
            time_window: Prediction window
            max_trials: Maximum number of trials (defaults to config)
            timeout: Timeout in seconds

        Returns:
            Optimization results dictionary
        """
        target_key = f"{flare_class}_{time_window}"

        self.logger.info(f"🚀 Starting optimization for {flare_class}-class, {time_window}h window")

        # Create study
        study = self.create_study(flare_class, time_window)
        self.studies[target_key] = study

        # Create objective function
        objective = create_objective(flare_class, time_window)

        # Determine number of trials
        if max_trials is None:
            max_trials = get_total_trials()

        start_time = time.time()

        try:
            # Run optimization
            study.optimize(
                objective,
                n_trials=max_trials,
                timeout=timeout,
                gc_after_trial=True,  # Garbage collection to save memory
                show_progress_bar=True
            )

            elapsed = time.time() - start_time

            # Analyze results
            results = self._analyze_study_results(study, flare_class, time_window, elapsed)
            self.results[target_key] = results

            # Save results
            self._save_study_results(study, results, flare_class, time_window)

            self.logger.info(f"✅ Optimization completed for {target_key}")
            self.logger.info(f"   Best TSS: {study.best_value:.4f}")
            self.logger.info(f"   Best params: {study.best_params}")
            self.logger.info(f"   Total time: {elapsed:.1f}s")

            return results

        except Exception as e:
            self.logger.error(f"❌ Optimization failed for {target_key}: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def run_all_targets(
        self,
        targets: Optional[List[Dict[str, str]]] = None,
        max_trials_per_target: Optional[int] = None,
        timeout_per_target: Optional[float] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run optimization for all target configurations.

        Args:
            targets: List of target configurations (defaults to config)
            max_trials_per_target: Max trials per target
            timeout_per_target: Timeout per target in seconds

        Returns:
            Results for all targets
        """
        if targets is None:
            targets = EXPERIMENT_TARGETS

        all_results = {}
        total_start = time.time()

        self.logger.info(f"🌟 Starting optimization for {len(targets)} target configurations")

        for i, target in enumerate(targets, 1):
            flare_class = target["flare_class"]
            time_window = target["time_window"]
            target_key = f"{flare_class}_{time_window}"

            self.logger.info(f"[{i}/{len(targets)}] Processing {target_key}")

            try:
                results = self.run_single_target(
                    flare_class,
                    time_window,
                    max_trials_per_target,
                    timeout_per_target
                )
                all_results[target_key] = results

            except Exception as e:
                self.logger.error(f"❌ Failed to optimize {target_key}: {e}")
                all_results[target_key] = {"error": str(e)}

        total_elapsed = time.time() - total_start

        # Save combined results
        self._save_combined_results(all_results, total_elapsed)

        self.logger.info(f"🏁 All optimizations completed in {total_elapsed:.1f}s")

        return all_results

    def _analyze_study_results(
        self,
        study: optuna.Study,
        flare_class: str,
        time_window: str,
        elapsed_time: float
    ) -> Dict[str, Any]:
        """Analyze and summarize study results."""

        # Basic study info
        results = {
            "study_name": study.study_name,
            "flare_class": flare_class,
            "time_window": time_window,
            "n_trials": len(study.trials),
            "optimization_time": elapsed_time,
            "timestamp": datetime.now().isoformat()
        }

        if len(study.trials) == 0:
            results["error"] = "No trials completed"
            return results

        # Best trial
        best_trial = study.best_trial
        results["best_trial"] = {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
            "user_attrs": best_trial.user_attrs
        }

        # Trial statistics by stage
        stage_stats = {}
        for trial in study.trials:
            stage = trial.user_attrs.get("stage", "unknown")
            if stage not in stage_stats:
                stage_stats[stage] = {
                    "trials": 0,
                    "completed": 0,
                    "pruned": 0,
                    "failed": 0,
                    "best_value": -float('inf')
                }

            stage_stats[stage]["trials"] += 1

            if trial.state == optuna.trial.TrialState.COMPLETE:
                stage_stats[stage]["completed"] += 1
                if trial.value > stage_stats[stage]["best_value"]:
                    stage_stats[stage]["best_value"] = trial.value
            elif trial.state == optuna.trial.TrialState.PRUNED:
                stage_stats[stage]["pruned"] += 1
            elif trial.state == optuna.trial.TrialState.FAIL:
                stage_stats[stage]["failed"] += 1

        results["stage_statistics"] = stage_stats

        # Hyperparameter importance (if enough trials)
        if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) >= 10:
            try:
                importance = optuna.importance.get_param_importances(study)
                results["param_importance"] = importance
            except BaseException:
                results["param_importance"] = {}
        else:
            results["param_importance"] = {}

        # Top N trials
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials:
            top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:10]
            results["top_trials"] = [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "user_attrs": t.user_attrs
                }
                for t in top_trials
            ]

        return results

    def _save_study_results(
        self,
        study: optuna.Study,
        results: Dict[str, Any],
        flare_class: str,
        time_window: str
    ) -> None:
        """Save study results to files."""

        # Create target-specific results directory
        target_dir = Path(OUTPUT_DIRS["results"]) / f"{flare_class}_{time_window}h"
        target_dir.mkdir(exist_ok=True)

        # Save results summary
        results_file = target_dir / "optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save trials dataframe
        df = study.trials_dataframe()
        if not df.empty:
            trials_file = target_dir / "trials.csv"
            df.to_csv(trials_file, index=False)

        # Save study object
        study_file = target_dir / "study.pkl"
        import pickle
        with open(study_file, 'wb') as f:
            pickle.dump(study, f)

        self.logger.info(f"Results saved to {target_dir}")

    def _save_combined_results(self, all_results: Dict[str, Any], total_time: float) -> None:
        """Save combined results across all targets."""

        combined = {
            "timestamp": datetime.now().isoformat(),
            "total_optimization_time": total_time,
            "git_info": self._get_git_info(),
            "config": {
                "optuna": OPTUNA_CONFIG,
                "stages": SEARCH_STAGES,
                "reproducibility": REPRODUCIBILITY_CONFIG
            },
            "results": all_results
        }

        # Save to main results file
        results_file = Path(OUTPUT_DIRS["results"]) / "hpo_combined_results.json"
        with open(results_file, 'w') as f:
            json.dump(combined, f, indent=2, default=str)

        # Create summary CSV
        summary_data = []
        for target_key, result in all_results.items():
            if "error" not in result and "best_trial" in result:
                row = {
                    "target": target_key,
                    "flare_class": result["flare_class"],
                    "time_window": result["time_window"],
                    "best_tss": result["best_trial"]["value"],
                    "n_trials": result["n_trials"],
                    "optimization_time": result["optimization_time"],
                    **result["best_trial"]["params"]
                }
                summary_data.append(row)

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = Path(OUTPUT_DIRS["results"]) / "hpo_summary.csv"
            summary_df.to_csv(summary_file, index=False)

        self.logger.info(f"Combined results saved to {results_file}")

    def _get_git_info(self) -> Dict[str, str]:
        """Get git information for reproducibility."""
        try:
            import subprocess
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
            return {"commit": commit, "branch": branch}
        except BaseException:
            return {"commit": "unknown", "branch": "unknown"}

    def get_study_summary(self, flare_class: str, time_window: str) -> Optional[Dict[str, Any]]:
        """Get summary of a completed study."""
        target_key = f"{flare_class}_{time_window}"
        return self.results.get(target_key)

    def list_completed_studies(self) -> List[str]:
        """List all completed study targets."""
        return list(self.results.keys())

    def export_best_configs(self) -> Dict[str, Dict[str, Any]]:
        """Export best configurations for all completed studies."""
        best_configs = {}

        for target_key, results in self.results.items():
            if "best_trial" in results:
                best_configs[target_key] = {
                    "params": results["best_trial"]["params"],
                    "tss": results["best_trial"]["value"],
                    "flare_class": results["flare_class"],
                    "time_window": results["time_window"]
                }

        return best_configs


def main():
    """Main function for running HPO studies."""
    import argparse

    parser = argparse.ArgumentParser(description="EVEREST Hyperparameter Optimization")
    parser.add_argument("--target", choices=["all", "single"], default="all",
                        help="Run all targets or single target")
    parser.add_argument("--flare-class", choices=["C", "M", "M5"], default="M",
                        help="Flare class for single target")
    parser.add_argument("--time-window", choices=["24", "48", "72"], default="24",
                        help="Time window for single target")
    parser.add_argument("--max-trials", type=int, default=None,
                        help="Maximum number of trials")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Timeout in seconds")

    args = parser.parse_args()

    # Create study manager
    manager = StudyManager()

    if args.target == "all":
        # Run all targets
        results = manager.run_all_targets(
            max_trials_per_target=args.max_trials,
            timeout_per_target=args.timeout
        )

        print("\n🎯 HPO Summary:")
        for target, result in results.items():
            if "best_trial" in result:
                print(f"  {target}: TSS = {result['best_trial']['value']:.4f}")
            else:
                print(f"  {target}: FAILED")

    else:
        # Run single target
        result = manager.run_single_target(
            args.flare_class,
            args.time_window,
            args.max_trials,
            args.timeout
        )

        print(f"\n🎯 Best result for {args.flare_class}-{args.time_window}h:")
        print(f"  TSS: {result['best_trial']['value']:.4f}")
        print(f"  Params: {result['best_trial']['params']}")


if __name__ == "__main__":
    main()
