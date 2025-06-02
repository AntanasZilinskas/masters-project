#!/usr/bin/env python3
"""
Run EVEREST Ablation Study with Updated Optimal Hyperparameters

This script runs the ablation study using the updated optimal hyperparameters:
- embed_dim: 64
- num_blocks: 8
- dropout: 0.23876978467047777
- focal_gamma: 3.4223204654921875
- learning_rate: 0.0006926769179941219
- batch_size: 1024

Usage:
    # Run all ablations (recommended)
    python run_updated_ablation.py

    # Run specific variants only
    python run_updated_ablation.py --variants full_model no_evidential no_evt

    # Run with fewer seeds for testing
    python run_updated_ablation.py --seeds 0 1

    # Skip sequence length study
    python run_updated_ablation.py --no-sequence-study

    # Run single-threaded for debugging
    python run_updated_ablation.py --max-workers 1
"""

from ablation.config import get_all_variant_names, RANDOM_SEEDS
from ablation.run_ablation_study import run_ablation_study
import os
import sys
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def main():
    """Main function to run ablation study with updated hyperparameters."""
    parser = argparse.ArgumentParser(description='Run EVEREST ablation study with updated hyperparameters')

    parser.add_argument('--variants', nargs='+',
                        choices=get_all_variant_names(),
                        help='Specific variants to run (default: all)')

    parser.add_argument('--seeds', nargs='+', type=int,
                        choices=RANDOM_SEEDS,
                        help='Specific seeds to run (default: all)')

    parser.add_argument('--no-sequence-study', action='store_true',
                        help='Skip sequence length ablation study')

    parser.add_argument('--max-workers', type=int, default=None,
                        help='Maximum number of parallel workers (default: auto)')

    parser.add_argument('--analysis-only', action='store_true',
                        help='Run analysis only (no training)')

    args = parser.parse_args()

    print("🔬 EVEREST Ablation Study - Updated Hyperparameters")
    print("=" * 60)
    print("📊 Updated Optimal Hyperparameters:")
    print("   • embed_dim: 64")
    print("   • num_blocks: 8")
    print("   • dropout: 0.239")
    print("   • focal_gamma: 3.422")
    print("   • learning_rate: 0.000693")
    print("   • batch_size: 1024")
    print()

    if args.analysis_only:
        print("📈 Running analysis only...")
        from ablation.analysis import AblationAnalyzer

        analyzer = AblationAnalyzer()
        analyzer.load_all_results()
        analyzer.aggregate_results()
        analyzer.perform_statistical_tests()
        analyzer.generate_visualizations()
        analyzer.save_summary_report()

        print("✅ Analysis complete!")
        return

    # Run the ablation study
    run_ablation_study(
        variants=args.variants,
        seeds=args.seeds,
        include_sequence_study=not args.no_sequence_study,
        max_workers=args.max_workers,
        run_analysis=True
    )

    print("\n🎉 Ablation study complete with updated hyperparameters!")
    print("\n📋 Next steps:")
    print("1. Check results in models/ablation/results/")
    print("2. Review plots in models/ablation/plots/")
    print("3. Examine statistical analysis output")
    print("4. Update thesis with new ablation results")


if __name__ == "__main__":
    main()
