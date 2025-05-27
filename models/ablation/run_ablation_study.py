#!/usr/bin/env python3
"""
Main script for running EVEREST ablation studies.

This script orchestrates the complete ablation study including:
- Component ablations (6 variants)
- Sequence length ablations (5 variants)
- 5 random seeds per variant
- Statistical analysis and visualization
"""

import os
import sys
import time
import argparse
import multiprocessing as mp
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ablation.config import (
    ABLATION_VARIANTS, SEQUENCE_LENGTH_VARIANTS, RANDOM_SEEDS,
    PRIMARY_TARGET, OUTPUT_CONFIG, get_all_variant_names,
    get_all_sequence_variants, validate_config
)
from ablation.trainer import train_ablation_variant
from ablation.analysis import AblationAnalyzer


def run_single_experiment(args: Tuple[str, int, Optional[str]]) -> dict:
    """
    Run a single ablation experiment.
    
    Args:
        args: Tuple of (variant_name, seed, sequence_variant)
        
    Returns:
        Experiment results dictionary
    """
    variant_name, seed, sequence_variant = args
    
    try:
        print(f"🔬 Starting experiment: {variant_name}, seed {seed}")
        if sequence_variant:
            print(f"   Sequence variant: {sequence_variant}")
        
        results = train_ablation_variant(variant_name, seed, sequence_variant)
        
        print(f"✅ Completed: {variant_name}, seed {seed}")
        print(f"   TSS: {results['final_metrics']['tss']:.4f}")
        print(f"   F1: {results['final_metrics']['f1']:.4f}")
        
        return {
            'status': 'success',
            'variant_name': variant_name,
            'seed': seed,
            'sequence_variant': sequence_variant,
            'results': results
        }
        
    except Exception as e:
        print(f"❌ Failed: {variant_name}, seed {seed}: {str(e)}")
        return {
            'status': 'failed',
            'variant_name': variant_name,
            'seed': seed,
            'sequence_variant': sequence_variant,
            'error': str(e)
        }


def generate_experiment_list(
    variants: List[str] = None,
    seeds: List[int] = None,
    include_sequence_study: bool = True
) -> List[Tuple[str, int, Optional[str]]]:
    """
    Generate list of all experiments to run.
    
    Args:
        variants: List of variant names to run (default: all)
        seeds: List of seeds to use (default: all)
        include_sequence_study: Whether to include sequence length study
        
    Returns:
        List of experiment tuples (variant_name, seed, sequence_variant)
    """
    if variants is None:
        variants = get_all_variant_names()
    
    if seeds is None:
        seeds = RANDOM_SEEDS
    
    experiments = []
    
    # Component ablation experiments
    for variant_name in variants:
        for seed in seeds:
            experiments.append((variant_name, seed, None))
    
    # Sequence length ablation experiments
    if include_sequence_study:
        for seq_variant in get_all_sequence_variants():
            for seed in seeds:
                # Use full model for sequence length study
                experiments.append(("full_model", seed, seq_variant))
    
    return experiments


def run_ablation_study(
    variants: List[str] = None,
    seeds: List[int] = None,
    include_sequence_study: bool = True,
    max_workers: int = None,
    run_analysis: bool = True
):
    """
    Run complete ablation study.
    
    Args:
        variants: List of variant names to run
        seeds: List of seeds to use
        include_sequence_study: Whether to include sequence length study
        max_workers: Maximum number of parallel workers
        run_analysis: Whether to run statistical analysis after training
    """
    print("🔬 EVEREST Ablation Study")
    print("=" * 50)
    print(f"Target: {PRIMARY_TARGET['flare_class']}-class, {PRIMARY_TARGET['time_window']}h")
    print()
    
    # Validate configuration
    validate_config()
    
    # Generate experiment list
    experiments = generate_experiment_list(variants, seeds, include_sequence_study)
    
    print(f"📊 Total experiments: {len(experiments)}")
    print(f"   Component ablations: {len(get_all_variant_names()) * len(RANDOM_SEEDS)}")
    if include_sequence_study:
        print(f"   Sequence length ablations: {len(get_all_sequence_variants()) * len(RANDOM_SEEDS)}")
    print(f"   Parallel workers: {max_workers or mp.cpu_count()}")
    print()
    
    # Run experiments
    start_time = time.time()
    results = []
    
    if max_workers == 1:
        # Sequential execution for debugging
        print("🔄 Running experiments sequentially...")
        for i, experiment in enumerate(experiments):
            print(f"\n[{i+1}/{len(experiments)}] Running experiment...")
            result = run_single_experiment(experiment)
            results.append(result)
    else:
        # Parallel execution
        print(f"🔄 Running experiments in parallel ({max_workers} workers)...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all experiments
            future_to_experiment = {
                executor.submit(run_single_experiment, exp): exp 
                for exp in experiments
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_experiment):
                result = future.result()
                results.append(result)
                completed += 1
                
                print(f"[{completed}/{len(experiments)}] Completed: "
                      f"{result['variant_name']}, seed {result['seed']}")
                
                if result['status'] == 'failed':
                    print(f"   ❌ Error: {result['error']}")
    
    # Summary
    elapsed = time.time() - start_time
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    print(f"\n🏁 Ablation study completed!")
    print(f"   Total time: {elapsed:.1f}s ({elapsed/3600:.2f}h)")
    print(f"   Successful: {successful}/{len(results)}")
    print(f"   Failed: {failed}/{len(results)}")
    
    if failed > 0:
        print(f"\n❌ Failed experiments:")
        for result in results:
            if result['status'] == 'failed':
                seq_info = f", {result['sequence_variant']}" if result['sequence_variant'] else ""
                print(f"   {result['variant_name']}, seed {result['seed']}{seq_info}: {result['error']}")
    
    # Run statistical analysis
    if run_analysis and successful > 0:
        print(f"\n📊 Running statistical analysis...")
        try:
            analyzer = AblationAnalyzer()
            analyzer.run_full_analysis()
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
    
    return results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Run EVEREST ablation studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full ablation study
  python run_ablation_study.py
  
  # Run only specific variants
  python run_ablation_study.py --variants full_model no_evidential no_evt
  
  # Run with specific seeds
  python run_ablation_study.py --seeds 0 1 2
  
  # Run sequentially for debugging
  python run_ablation_study.py --max-workers 1
  
  # Skip sequence length study
  python run_ablation_study.py --no-sequence-study
  
  # Only run analysis (no training)
  python run_ablation_study.py --analysis-only
        """
    )
    
    parser.add_argument(
        "--variants", 
        nargs="+", 
        choices=get_all_variant_names(),
        help="Specific variants to run (default: all)"
    )
    
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=RANDOM_SEEDS,
        help=f"Random seeds to use (default: {RANDOM_SEEDS})"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: CPU count)"
    )
    
    parser.add_argument(
        "--no-sequence-study",
        action="store_true",
        help="Skip sequence length ablation study"
    )
    
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip statistical analysis after training"
    )
    
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Only run analysis (skip training)"
    )
    
    args = parser.parse_args()
    
    if args.analysis_only:
        print("📊 Running analysis only...")
        analyzer = AblationAnalyzer()
        analyzer.run_full_analysis()
        return
    
    # Run ablation study
    results = run_ablation_study(
        variants=args.variants,
        seeds=args.seeds,
        include_sequence_study=not args.no_sequence_study,
        max_workers=args.max_workers,
        run_analysis=not args.no_analysis
    )
    
    print(f"\n✅ Ablation study complete!")
    print(f"📁 Results saved to: {OUTPUT_CONFIG['results_dir']}")
    print(f"📊 Plots saved to: {OUTPUT_CONFIG['plots_dir']}")


if __name__ == "__main__":
    main() 