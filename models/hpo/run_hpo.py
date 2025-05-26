#!/usr/bin/env python3
"""
EVEREST Hyperparameter Optimization Runner

This script provides a convenient interface to run hyperparameter optimization
studies using the three-tier Bayesian search framework.

Usage examples:
    # Run optimization for all 9 target configurations
    python run_hpo.py --target all
    
    # Run optimization for a single target  
    python run_hpo.py --target single --flare-class M --time-window 24
    
    # Quick test with limited trials
    python run_hpo.py --target single --flare-class M --time-window 24 --max-trials 5
    
    # Run with timeout (useful for CI/CD)
    python run_hpo.py --target all --timeout 3600  # 1 hour per target
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.hpo import StudyManager, HPO_SEARCH_SPACE, SEARCH_STAGES, EXPERIMENT_TARGETS


def print_banner():
    """Print welcome banner."""
    print("=" * 80)
    print("🚀 EVEREST Hyperparameter Optimization Framework")
    print("   Three-tier Bayesian Search using Optuna v3.6")
    print("=" * 80)
    print(f"📊 Search space: {len(HPO_SEARCH_SPACE)} parameters")
    print(f"🔄 Optimization stages: {list(SEARCH_STAGES.keys())}")
    print(f"🎯 Target configurations: {len(EXPERIMENT_TARGETS)}")
    print("=" * 80)


def print_config_summary():
    """Print configuration summary."""
    total_trials = sum(stage["trials"] for stage in SEARCH_STAGES.values())
    print("\n📋 Configuration Summary:")
    print(f"   • Total trials per target: {total_trials}")
    print(f"   • Exploration: {SEARCH_STAGES['exploration']['trials']} trials × {SEARCH_STAGES['exploration']['epochs']} epochs")
    print(f"   • Refinement: {SEARCH_STAGES['refinement']['trials']} trials × {SEARCH_STAGES['refinement']['epochs']} epochs") 
    print(f"   • Confirmation: {SEARCH_STAGES['confirmation']['trials']} trials × {SEARCH_STAGES['confirmation']['epochs']} epochs")
    print()


def run_single_target(args):
    """Run optimization for a single target."""
    print(f"\n🎯 Running optimization for {args.flare_class}-class, {args.time_window}h window")
    
    if args.max_trials:
        print(f"   • Limited to {args.max_trials} trials")
    if args.timeout:
        print(f"   • Timeout: {args.timeout}s ({args.timeout/3600:.1f}h)")
    
    manager = StudyManager()
    
    try:
        result = manager.run_single_target(
            args.flare_class,
            args.time_window,
            args.max_trials,
            args.timeout
        )
        
        print(f"\n🎉 Optimization completed!")
        print(f"   • Best TSS: {result['best_trial']['value']:.4f}")
        print(f"   • Best parameters: {result['best_trial']['params']}")
        print(f"   • Total trials: {result['n_trials']}")
        print(f"   • Optimization time: {result['optimization_time']:.1f}s")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        return False


def run_all_targets(args):
    """Run optimization for all targets."""
    print(f"\n🌟 Running optimization for all {len(EXPERIMENT_TARGETS)} target configurations")
    
    if args.max_trials:
        print(f"   • Limited to {args.max_trials} trials per target")
    if args.timeout:
        print(f"   • Timeout: {args.timeout}s ({args.timeout/3600:.1f}h) per target")
    
    manager = StudyManager()
    
    try:
        results = manager.run_all_targets(
            max_trials_per_target=args.max_trials,
            timeout_per_target=args.timeout
        )
        
        print(f"\n🎉 All optimizations completed!")
        print("\n📊 Summary Results:")
        
        successful = 0
        for target_key, result in results.items():
            if "best_trial" in result:
                tss = result["best_trial"]["value"]
                trials = result["n_trials"]
                time_taken = result["optimization_time"]
                print(f"   • {target_key:8}: TSS={tss:.4f}, trials={trials:3d}, time={time_taken:6.1f}s")
                successful += 1
            else:
                print(f"   • {target_key:8}: FAILED - {result.get('error', 'unknown error')}")
        
        print(f"\n✅ Success rate: {successful}/{len(EXPERIMENT_TARGETS)} ({100*successful/len(EXPERIMENT_TARGETS):.1f}%)")
        
        return successful > 0
        
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="EVEREST Hyperparameter Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--target", 
        choices=["all", "single"], 
        default="single",
        help="Run all targets or single target (default: single)"
    )
    
    parser.add_argument(
        "--flare-class", 
        choices=["C", "M", "M5"], 
        default="M",
        help="Flare class for single target (default: M)"
    )
    
    parser.add_argument(
        "--time-window", 
        choices=["24", "48", "72"], 
        default="24",
        help="Time window for single target (default: 24)"
    )
    
    parser.add_argument(
        "--max-trials", 
        type=int, 
        default=None,
        help="Maximum number of trials (default: use full 3-stage protocol)"
    )
    
    parser.add_argument(
        "--timeout", 
        type=float, 
        default=None,
        help="Timeout in seconds per target (default: no timeout)"
    )
    
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Suppress banner and configuration summary"
    )
    
    args = parser.parse_args()
    
    if not args.quiet:
        print_banner()
        print_config_summary()
    
    # Run optimization
    if args.target == "all":
        success = run_all_targets(args)
    else:
        success = run_single_target(args)
    
    if success:
        print("\n🚀 HPO completed successfully!")
        print("📁 Results saved to models/hpo/results/")
        print("📊 Visualizations saved to models/hpo/plots/")
        sys.exit(0)
    else:
        print("\n💥 HPO failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 