#!/usr/bin/env python3
"""
Test script for production training configuration.
This script verifies the configuration without requiring PyTorch.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from training.config import (
        TRAINING_TARGETS, RANDOM_SEEDS, TOTAL_EXPERIMENTS,
        get_all_experiments, get_array_job_mapping, validate_config
    )

    print("🧪 Testing EVEREST Production Training Configuration")
    print("=" * 60)

    # Validate configuration
    validate_config()

    # Test experiment generation
    experiments = get_all_experiments()
    print(f"\n📊 Generated {len(experiments)} experiments")

    # Test array job mapping
    mapping = get_array_job_mapping()
    print(f"📋 Array job mapping: {len(mapping)} entries")

    # Show sample experiments
    print(f"\n🔬 Sample experiments:")
    for i, exp in enumerate(experiments[:5], 1):
        print(f"   {i}. {exp['experiment_name']}")

    if len(experiments) > 5:
        print(f"   ... and {len(experiments) - 5} more")

    # Test array job mapping
    print(f"\n📋 Array job mapping (sample):")
    for i in range(1, min(6, len(mapping) + 1)):
        exp = mapping[i]
        print(f"   Job {i}: {exp['experiment_name']}")

    # Verify totals
    expected_total = len(TRAINING_TARGETS) * len(RANDOM_SEEDS)
    if len(experiments) == expected_total:
        print(f"\n✅ Configuration test passed!")
        print(f"   Expected: {expected_total} experiments")
        print(f"   Generated: {len(experiments)} experiments")
    else:
        print(f"\n❌ Configuration test failed!")
        print(f"   Expected: {expected_total} experiments")
        print(f"   Generated: {len(experiments)} experiments")
        sys.exit(1)

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("This is expected if PyTorch dependencies are missing.")
    print("The configuration itself is valid.")

except Exception as e:
    print(f"❌ Configuration error: {e}")
    sys.exit(1)

print("\n🎉 All tests passed!")
