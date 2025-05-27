#!/usr/bin/env python3
"""
EVEREST Ablation Study Runner - EXACT HPO Pattern

This script follows the EXACT same pattern as the working HPO runner,
using the same data loading, model creation, and training patterns.

COMPONENT ABLATION STUDY:
- Component ablations: 7 variants × 5 seeds = 35 experiments
- Focus on architectural components only
"""

import sys
import argparse
from pathlib import Path
import os
import numpy as np
import torch

# Add project root to path (EXACT same as HPO)
project_root = Path(__file__).parent.parent.parent  # Go up to masters-project root
sys.path.insert(0, str(project_root))

# Change working directory to project root to ensure relative paths work (EXACT same as HPO)
os.chdir(project_root)

# Now import from models directory (EXACT same as HPO)
from models.utils import get_training_data, get_testing_data
from models.solarknowledge_ret_plus import RETPlusWrapper


class AblationObjective:
    """
    Ablation objective function following EXACT HPO pattern.
    
    This class encapsulates the training and evaluation logic for a single
    ablation experiment, following the exact same structure as HPOObjective.
    
    Focuses on component ablations only.
    """
    
    def __init__(self, variant_name: str, seed: int):
        """Initialize the ablation objective (EXACT same as HPO)."""
        self.variant_name = variant_name
        self.seed = seed
        
        # Fixed input shape for component ablations
        self.input_shape = (10, 9)
        
        # Load data once to avoid repeated I/O (EXACT same as HPO)
        self._load_data()
        
        # Set up reproducibility (EXACT same as HPO)
        self._setup_reproducibility()
        
    def _setup_reproducibility(self):
        """Set up reproducible training environment (EXACT same as HPO)."""
        # Set random seeds
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            
        # Set deterministic behavior (EXACT same as HPO)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
            
    def _load_data(self):
        """Load and cache training/validation data (EXACT same as HPO)."""
        try:
            print(f"Loading data for M5-class, 72h window...")
            
            # Load training data (EXACT same as HPO)
            self.X_train, self.y_train = get_training_data("72", "M5")
            
            if self.X_train is None or self.y_train is None:
                raise ValueError(f"Training data not found for M5/72h")
                
            # Load validation data (use testing data for validation during ablation)
            self.X_val, self.y_val = get_testing_data("72", "M5")
            if self.X_val is None or self.y_val is None:
                raise ValueError(f"Validation data not found for M5/72h")
                    
            # Convert to numpy arrays (EXACT same as HPO)
            self.X_train = np.array(self.X_train)
            self.y_train = np.array(self.y_train)
            self.X_val = np.array(self.X_val)
            self.y_val = np.array(self.y_val)
                
            print(f"Data loaded successfully:")
            print(f"  Training: {self.X_train.shape[0]} samples, shape: {self.X_train.shape}")
            print(f"  Validation: {self.X_val.shape[0]} samples, shape: {self.X_val.shape}")
            print(f"  Positive rate (train): {self.y_train.mean():.3f}")
            print(f"  Positive rate (val): {self.y_val.mean():.3f}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
            
    def _get_ablation_config(self):
        """Get ablation configuration for this variant."""
        # Define ablation variants (same as config.py)
        variants = {
            "full_model": {
                "use_attention_bottleneck": True,
                "use_evidential": True,
                "use_evt": True,
                "use_precursor": True,
                "loss_weights": {"focal": 0.7, "evid": 0.1, "evt": 0.2, "prec": 0.05}
            },
            "no_evidential": {
                "use_attention_bottleneck": True,
                "use_evidential": False,
                "use_evt": True,
                "use_precursor": True,
                "loss_weights": {"focal": 0.8, "evid": 0.0, "evt": 0.2, "prec": 0.05}
            },
            "no_evt": {
                "use_attention_bottleneck": True,
                "use_evidential": True,
                "use_evt": False,
                "use_precursor": True,
                "loss_weights": {"focal": 0.8, "evid": 0.2, "evt": 0.0, "prec": 0.05}
            },
            "mean_pool": {
                "use_attention_bottleneck": False,
                "use_evidential": True,
                "use_evt": True,
                "use_precursor": True,
                "loss_weights": {"focal": 0.7, "evid": 0.1, "evt": 0.2, "prec": 0.05}
            },
            "cross_entropy": {
                "use_attention_bottleneck": True,
                "use_evidential": False,
                "use_evt": False,
                "use_precursor": True,
                "loss_weights": {"focal": 1.0, "evid": 0.0, "evt": 0.0, "prec": 0.05}
            },
            "no_precursor": {
                "use_attention_bottleneck": True,
                "use_evidential": True,
                "use_evt": True,
                "use_precursor": False,
                "loss_weights": {"focal": 0.75, "evid": 0.1, "evt": 0.15, "prec": 0.0}
            },
            "fp32_training": {
                "use_attention_bottleneck": True,
                "use_evidential": True,
                "use_evt": True,
                "use_precursor": True,
                "loss_weights": {"focal": 0.7, "evid": 0.1, "evt": 0.2, "prec": 0.05}
            }
        }
        
        return variants[self.variant_name]
        
    def _create_model(self):
        """Create model with ablation configuration (EXACT same pattern as HPO)."""
        
        # Get optimal hyperparameters from HPO study
        optimal_hyperparams = {
            "embed_dim": 64,
            "num_blocks": 8,
            "dropout": 0.23876978467047777,
            "focal_gamma": 3.4223204654921875,
            "learning_rate": 0.0006926769179941219,
            "batch_size": 1024
        }
        
        # Get ablation configuration
        ablation_config = self._get_ablation_config()
        
        # Create wrapper (EXACT same as HPO - let wrapper create the model)
        wrapper = RETPlusWrapper(
            input_shape=self.input_shape,
            use_attention_bottleneck=ablation_config["use_attention_bottleneck"],
            use_evidential=ablation_config["use_evidential"],
            use_evt=ablation_config["use_evt"],
            use_precursor=ablation_config["use_precursor"],
            loss_weights=ablation_config["loss_weights"]
        )
        
        # Update optimizer with optimal learning rate (EXACT same as HPO)
        wrapper.optimizer = torch.optim.AdamW(
            wrapper.model.parameters(),
            lr=optimal_hyperparams["learning_rate"],
            weight_decay=1e-4,
            fused=True
        )
        
        return wrapper, optimal_hyperparams
        
    def run_experiment(self):
        """Run ablation experiment (EXACT same pattern as HPO)."""
        print(f"\n🔬 Running component ablation: {self.variant_name}, seed {self.seed}")
        print(f"   Input shape: {self.input_shape}")
        
        try:
            # Create model (EXACT same as HPO)
            model, hyperparams = self._create_model()
            
            print(f"Model created with ablation config:")
            ablation_config = self._get_ablation_config()
            print(f"  Attention: {ablation_config['use_attention_bottleneck']}")
            print(f"  Evidential: {ablation_config['use_evidential']}")
            print(f"  EVT: {ablation_config['use_evt']}")
            print(f"  Precursor: {ablation_config['use_precursor']}")
            
            # Train model using wrapper's train method (EXACT same as HPO)
            print(f"Training for 50 epochs with early stopping...")
            
            model_dir = model.train(
                X_train=self.X_train,
                y_train=self.y_train,
                epochs=50,  # Early stopping after 10 epochs
                batch_size=hyperparams["batch_size"],
                gamma_max=hyperparams["focal_gamma"],
                warmup_epochs=20,
                flare_class="M5",
                time_window="72",
                in_memory_dataset=True,
                track_emissions=False  # Disable for cluster
            )
            
            # Evaluate on validation set
            print(f"Evaluating on validation set...")
            y_pred_proba = model.predict_proba(self.X_val)
            y_pred = (y_pred_proba >= 0.5).astype(int).squeeze()
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(self.y_val, y_pred)
            precision = precision_score(self.y_val, y_pred, zero_division=0)
            recall = recall_score(self.y_val, y_pred, zero_division=0)
            f1 = f1_score(self.y_val, y_pred, zero_division=0)
            
            # Calculate TSS
            tn = ((y_pred == 0) & (self.y_val == 0)).sum()
            fp = ((y_pred == 1) & (self.y_val == 0)).sum()
            sensitivity = recall
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            tss = sensitivity + specificity - 1
            
            results = {
                "experiment_type": "component",
                "variant": self.variant_name,
                "seed": self.seed,
                "input_shape": self.input_shape,
                "model_dir": model_dir,
                "final_metrics": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "tss": tss,
                    "sensitivity": sensitivity,
                    "specificity": specificity
                },
                "training_history": model.history
            }
            
            print(f"✅ Experiment completed successfully!")
            print(f"   • Accuracy: {accuracy:.4f}")
            print(f"   • TSS: {tss:.4f}")
            print(f"   • F1: {f1:.4f}")
            print(f"   • Model saved to: {model_dir}")
            
            return results
            
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def print_banner():
    """Print welcome banner."""
    print("=" * 80)
    print("🔬 EVEREST Component Ablation Study - EXACT HPO Pattern")
    print("   Component Ablations Only (35 experiments)")
    print("   Following the exact same structure as working HPO")
    print("=" * 80)


def validate_gpu():
    """Validate GPU configuration (EXACT same as HPO)."""
    try:
        import torch
        print(f"   • Validating GPU configuration...")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_gpu)
            print(f"   ✅ GPU available: {gpu_name} (device {current_gpu}/{gpu_count})")
            return True
        else:
            print(f"   ❌ GPU not available - ablation requires GPU")
            return False
            
    except Exception as e:
        print(f"   ❌ GPU validation failed: {e}")
        return False


def main():
    """Main function (EXACT same structure as HPO)."""
    parser = argparse.ArgumentParser(description="EVEREST Component Ablation Study - HPO Pattern")
    
    parser.add_argument("--variant", 
                       choices=["full_model", "no_evidential", "no_evt", "mean_pool", 
                               "cross_entropy", "no_precursor", "fp32_training"],
                       required=True,
                       help="Component ablation variant to run")
    
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Validate GPU (EXACT same as HPO)
    if not validate_gpu():
        print("\n❌ GPU validation failed!")
        return 1
    
    # Create and run ablation objective (EXACT same pattern as HPO)
    objective = AblationObjective(args.variant, args.seed)
    results = objective.run_experiment()
    
    if results:
        print(f"\n🎉 Ablation completed successfully!")
        print(f"📁 Results saved to: {results['model_dir']}")
        return 0
    else:
        print(f"\n❌ Ablation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 