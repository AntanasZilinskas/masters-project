"""
EVEREST Production Training Orchestration

This script orchestrates the training of all production EVEREST models
across all flare class × time window combinations with multiple seeds.
"""

import os
import sys
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Robust import handling for both module and script execution
try:
    # Relative imports (when used as module)
    from .config import (
        TRAINING_TARGETS, RANDOM_SEEDS, TOTAL_EXPERIMENTS, OUTPUT_CONFIG,
        get_all_experiments, get_array_job_mapping, create_output_directories
    )
    from .trainer import train_production_model
except ImportError:
    # Absolute imports (when used as script from project root)
    try:
        from models.training.config import (
            TRAINING_TARGETS, RANDOM_SEEDS, TOTAL_EXPERIMENTS, OUTPUT_CONFIG,
            get_all_experiments, get_array_job_mapping, create_output_directories
        )
        from models.training.trainer import train_production_model
    except ImportError:
        # Direct imports (when script run from training directory)
        from config import (
            TRAINING_TARGETS, RANDOM_SEEDS, TOTAL_EXPERIMENTS, OUTPUT_CONFIG,
            get_all_experiments, get_array_job_mapping, create_output_directories
        )
        from trainer import train_production_model


def run_single_experiment(experiment_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single training experiment."""
    try:
        print(f"🚀 Starting {experiment_config['experiment_name']}")

        results = train_production_model(
            flare_class=experiment_config["flare_class"],
            time_window=experiment_config["time_window"],
            seed=experiment_config["seed"]
        )

        print(f"✅ Completed {experiment_config['experiment_name']}")
        print(f"   TSS: {results['final_metrics']['tss']:.4f}")
        print(f"   Threshold: {results['optimal_threshold']:.3f}")

        return {
            "status": "success",
            "experiment_name": experiment_config["experiment_name"],
            "results": results
        }

    except Exception as e:
        print(f"❌ Failed {experiment_config['experiment_name']}: {str(e)}")
        return {
            "status": "failed",
            "experiment_name": experiment_config["experiment_name"],
            "error": str(e)
        }


def run_array_job_experiment(array_index: int) -> Dict[str, Any]:
    """Run experiment based on PBS array job index."""
    mapping = get_array_job_mapping()

    if array_index not in mapping:
        raise ValueError(f"Invalid array index: {array_index}")

    experiment_config = mapping[array_index]
    return run_single_experiment(experiment_config)


def run_all_experiments(max_workers: int = 1, targets_filter: List[str] = None) -> Dict[str, Any]:
    """Run all production training experiments."""
    print("🏭 Starting EVEREST Production Training")
    print("=" * 60)

    # Create output directories
    create_output_directories()

    # Get experiments to run
    all_experiments = get_all_experiments()

    # Filter experiments if specified
    if targets_filter:
        filtered_experiments = []
        for exp in all_experiments:
            target_key = f"{exp['flare_class']}-{exp['time_window']}"
            if target_key in targets_filter:
                filtered_experiments.append(exp)
        all_experiments = filtered_experiments

    print(f"📊 Running {len(all_experiments)} experiments")
    print(f"🔧 Max workers: {max_workers}")

    if targets_filter:
        print(f"🎯 Filtered targets: {targets_filter}")

    start_time = time.time()
    results = []

    if max_workers == 1:
        # Sequential execution
        for i, experiment in enumerate(all_experiments, 1):
            print(f"\n[{i}/{len(all_experiments)}] {experiment['experiment_name']}")
            result = run_single_experiment(experiment)
            results.append(result)
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_experiment = {
                executor.submit(run_single_experiment, exp): exp
                for exp in all_experiments
            }

            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_experiment), 1):
                experiment = future_to_experiment[future]
                print(f"\n[{i}/{len(all_experiments)}] Collecting {experiment['experiment_name']}")

                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"❌ Exception in {experiment['experiment_name']}: {str(e)}")
                    results.append({
                        "status": "failed",
                        "experiment_name": experiment["experiment_name"],
                        "error": str(e)
                    })

    total_time = time.time() - start_time

    # Summarize results
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    print(f"\n📊 Production Training Summary")
    print("=" * 60)
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_time:.1f}s ({total_time/3600:.1f}h)")

    if failed:
        print(f"\n❌ Failed experiments:")
        for result in failed:
            print(f"   {result['experiment_name']}: {result['error']}")

    # Save summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_experiments": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "total_time": total_time,
        "results": results
    }

    summary_file = os.path.join(OUTPUT_CONFIG["results_dir"], "training_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n📁 Summary saved to: {summary_file}")

    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="EVEREST Production Training")

    # Execution mode
    parser.add_argument("--mode", choices=["all", "single", "array"], default="all",
                        help="Execution mode")

    # Single experiment parameters
    parser.add_argument("--flare_class", choices=["C", "M", "M5"],
                        help="Flare class for single experiment")
    parser.add_argument("--time_window", choices=["24", "48", "72"],
                        help="Time window for single experiment")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for single experiment")

    # Array job parameter
    parser.add_argument("--array_index", type=int,
                        help="PBS array job index (1-45)")

    # Parallel execution
    parser.add_argument("--max_workers", type=int, default=1,
                        help="Maximum parallel workers")

    # Filtering
    parser.add_argument("--targets", nargs="+",
                        help="Filter targets (e.g., C-24 M-48 M5-72)")

    # Dry run
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be executed without running")

    args = parser.parse_args()

    if args.dry_run:
        print("🔍 DRY RUN - Commands that would be executed:")

        if args.mode == "all":
            experiments = get_all_experiments()
            if args.targets:
                experiments = [exp for exp in experiments
                               if f"{exp['flare_class']}-{exp['time_window']}" in args.targets]

            print(f"Would run {len(experiments)} experiments:")
            for exp in experiments[:5]:  # Show first 5
                print(f"   {exp['experiment_name']}")
            if len(experiments) > 5:
                print(f"   ... and {len(experiments) - 5} more")

        elif args.mode == "single":
            if not all([args.flare_class, args.time_window]):
                print("Error: --flare_class and --time_window required for single mode")
                return
            print(f"Would run: {args.flare_class}-{args.time_window}h seed {args.seed}")

        elif args.mode == "array":
            if args.array_index is None:
                print("Error: --array_index required for array mode")
                return
            mapping = get_array_job_mapping()
            if args.array_index in mapping:
                exp = mapping[args.array_index]
                print(f"Would run array job {args.array_index}: {exp['experiment_name']}")
            else:
                print(f"Error: Invalid array index {args.array_index}")

        return

    # Execute based on mode
    if args.mode == "single":
        if not all([args.flare_class, args.time_window]):
            print("Error: --flare_class and --time_window required for single mode")
            return

        print(f"🏭 Training single model: {args.flare_class}-{args.time_window}h seed {args.seed}")

        results = train_production_model(args.flare_class, args.time_window, args.seed)

        print(f"\n🎯 Results:")
        print(f"   TSS: {results['final_metrics']['tss']:.4f}")
        print(f"   F1: {results['final_metrics']['f1']:.4f}")
        print(f"   Threshold: {results['optimal_threshold']:.3f}")
        print(f"   Latency: {results['final_metrics']['latency_ms']:.1f} ms")

    elif args.mode == "array":
        if args.array_index is None:
            print("Error: --array_index required for array mode")
            return

        print(f"🏭 Running array job {args.array_index}")

        result = run_array_job_experiment(args.array_index)

        if result["status"] == "success":
            print(f"✅ Array job {args.array_index} completed successfully")
        else:
            print(f"❌ Array job {args.array_index} failed: {result['error']}")
            sys.exit(1)

    elif args.mode == "all":
        summary = run_all_experiments(
            max_workers=args.max_workers,
            targets_filter=args.targets
        )

        if summary["failed"] > 0:
            print(f"⚠️  {summary['failed']} experiments failed")
            sys.exit(1)
        else:
            print("🎉 All experiments completed successfully!")


if __name__ == "__main__":
    main()
