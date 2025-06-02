"""
Quick test to verify ablation study saving mechanism
"""

from trainer import AblationTrainer
import os
import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def test_ablation_saving():
    """Test ablation study with minimal training to verify saving."""
    print("🧪 Testing ablation study saving mechanism...")

    # Override config for quick test
    from config import TRAINING_CONFIG, OPTIMAL_HYPERPARAMS

    # Temporarily modify config for ultra-fast testing
    original_epochs = TRAINING_CONFIG["epochs"]
    original_batch_size = OPTIMAL_HYPERPARAMS["batch_size"]

    TRAINING_CONFIG["epochs"] = 3  # Just 3 epochs for testing
    OPTIMAL_HYPERPARAMS["batch_size"] = 128  # Smaller batch for speed

    try:
        # Create trainer
        trainer = AblationTrainer("full_model", seed=99)  # Use seed 99 for testing

        print(f"📁 Expected save locations:")
        print(f"   Results: {trainer.experiment_dir}")
        print(f"   Models: {trainer.model_dir}")
        print(f"   Logs: {trainer.log_dir}")

        # Run quick training
        print(f"\n🚀 Starting quick test training...")
        start_time = time.time()

        results = trainer.train(batch_size_override=128, memory_efficient=True)

        elapsed = time.time() - start_time
        print(f"\n✅ Test completed in {elapsed:.1f}s")

        # Check what was saved
        print(f"\n📊 Checking saved files...")

        # Check results directory
        results_files = list(Path(trainer.experiment_dir).glob("*"))
        print(f"   Results directory ({len(results_files)} files):")
        for f in results_files:
            print(f"     📄 {f.name}")

        # Check models directory
        model_files = list(Path(trainer.model_dir).glob("*"))
        print(f"   Models directory ({len(model_files)} files):")
        for f in model_files:
            print(f"     💾 {f.name}")

        # Check isolated models (from trainer._save_model_weights)
        if "model_save_path" in results:
            isolated_path = Path(results["model_save_path"])
            if isolated_path.exists():
                isolated_files = list(isolated_path.glob("*"))
                print(f"   Isolated model directory ({len(isolated_files)} files):")
                for f in isolated_files:
                    print(f"     🔐 {f.name}")

        print(f"\n🎯 Final metrics:")
        for key, value in results["final_metrics"].items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.4f}")

        return results

    finally:
        # Restore original config
        TRAINING_CONFIG["epochs"] = original_epochs
        OPTIMAL_HYPERPARAMS["batch_size"] = original_batch_size


if __name__ == "__main__":
    test_ablation_saving()
