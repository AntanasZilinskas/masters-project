#!/usr/bin/env python3
"""
EVEREST Component Ablation Study with Enhanced Metadata Tracking

This version properly saves ablation variant and seed information in model metadata
to distinguish between different ablation experiments.
"""

import sys
import os
import argparse
import numpy as np
import torch
import random
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.solarknowledge_ret_plus import RETPlusWrapper
from models.utils import get_training_data, get_testing_data


class AblationObjectiveWithMetadata:
    """Enhanced ablation objective that properly tracks metadata."""
    
    def __init__(self, variant_name: str, seed: int):
        self.variant_name = variant_name
        self.seed = seed
        self.input_shape = (10, 9)
        
        # Setup reproducibility
        self._setup_reproducibility()
        
        # Load data once (EXACT same as HPO pattern)
        print(f"Loading data for ablation study...")
        self._load_data()
        
    def _setup_reproducibility(self):
        """Setup reproducible training (EXACT same as HPO)."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def _load_data(self):
        """Load training and validation data (EXACT same as HPO)."""
        try:
            # Load training data
            self.X_train, self.y_train = get_training_data('72', 'M5')
            if self.X_train is None or self.y_train is None:
                raise ValueError("Training data not found")
            
            # Load validation data  
            self.X_val, self.y_val = get_testing_data('72', 'M5')
            if self.X_val is None or self.y_val is None:
                raise ValueError("Validation data not found")
                
            print(f"✅ Data loaded: {len(self.X_train)} train, {len(self.X_val)} val samples")
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            raise
            
    def _get_ablation_config(self):
        """Get ablation configuration for the specified variant."""
        variants = {
            "full_model": {
                "use_attention_bottleneck": True,
                "use_evidential": True,
                "use_evt": True,
                "use_precursor": True,
                "loss_weights": {"focal": 0.8, "evid": 0.1, "evt": 0.1, "prec": 0.05},
                "description": "Full EVEREST model with all components (baseline)"
            },
            "no_evidential": {
                "use_attention_bottleneck": True,
                "use_evidential": False,
                "use_evt": True,
                "use_precursor": True,
                "loss_weights": {"focal": 0.9, "evid": 0.0, "evt": 0.1, "prec": 0.05},
                "description": "EVEREST model without evidential uncertainty (NIG head removed)"
            },
            "no_evt": {
                "use_attention_bottleneck": True,
                "use_evidential": True,
                "use_evt": False,
                "use_precursor": True,
                "loss_weights": {"focal": 0.8, "evid": 0.2, "evt": 0.0, "prec": 0.05},
                "description": "EVEREST model without EVT tail modeling (GPD head removed)"
            },
            "mean_pool": {
                "use_attention_bottleneck": False,
                "use_evidential": True,
                "use_evt": True,
                "use_precursor": True,
                "loss_weights": {"focal": 0.8, "evid": 0.1, "evt": 0.1, "prec": 0.05},
                "description": "EVEREST model with mean pooling instead of attention pooling"
            },
            "cross_entropy": {
                "use_attention_bottleneck": True,
                "use_evidential": False,
                "use_evt": False,
                "use_precursor": True,
                "loss_weights": {"focal": 0.0, "evid": 0.0, "evt": 0.0, "prec": 0.05},
                "description": "EVEREST model with standard cross-entropy loss (no focal/evidential/EVT)"
            },
            "no_precursor": {
                "use_attention_bottleneck": True,
                "use_evidential": True,
                "use_evt": True,
                "use_precursor": False,
                "loss_weights": {"focal": 0.8, "evid": 0.1, "evt": 0.1, "prec": 0.0},
                "description": "EVEREST model without precursor prediction head"
            },
            "fp32_training": {
                "use_attention_bottleneck": True,
                "use_evidential": True,
                "use_evt": True,
                "use_precursor": True,
                "loss_weights": {"focal": 0.8, "evid": 0.1, "evt": 0.1, "prec": 0.05},
                "description": "EVEREST model trained with FP32 precision (no mixed precision)"
            }
        }
        
        return variants[self.variant_name]
        
    def _create_enhanced_wrapper(self):
        """Create model wrapper with enhanced metadata tracking."""
        
        # Get optimal hyperparameters from HPO study
        optimal_hyperparams = {
            "embed_dim": 128,
            "num_blocks": 4,
            "dropout": 0.3531616510212273,
            "focal_gamma": 2.8033450352296265,
            "learning_rate": 0.0005337429672856022,
            "batch_size": 512
        }
        
        # Get ablation configuration
        ablation_config = self._get_ablation_config()
        
        # Create model with optimal hyperparameters
        from models.solarknowledge_ret_plus import RETPlusModel
        import torch
        
        model = RETPlusModel(
            input_shape=self.input_shape,
            embed_dim=optimal_hyperparams["embed_dim"],
            num_heads=4,  # Keep fixed
            ff_dim=optimal_hyperparams["embed_dim"] * 2,  # Scale with embed_dim
            num_blocks=optimal_hyperparams["num_blocks"],
            dropout=optimal_hyperparams["dropout"],
            use_attention_bottleneck=ablation_config["use_attention_bottleneck"],
            use_evidential=ablation_config["use_evidential"],
            use_evt=ablation_config["use_evt"],
            use_precursor=ablation_config["use_precursor"]
        )
        
        # Create wrapper and manually set the model
        wrapper = RETPlusWrapper(
            input_shape=self.input_shape,
            use_attention_bottleneck=ablation_config["use_attention_bottleneck"],
            use_evidential=ablation_config["use_evidential"],
            use_evt=ablation_config["use_evt"],
            use_precursor=ablation_config["use_precursor"],
            loss_weights=ablation_config["loss_weights"]
        )
        
        # Replace the default model with our optimized one
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        wrapper.model = model.to(device)
        
        # Update optimizer with optimal learning rate
        wrapper.optimizer = torch.optim.AdamW(
            wrapper.model.parameters(),
            lr=optimal_hyperparams["learning_rate"],
            weight_decay=1e-4,
            fused=True
        )
        
        return wrapper, optimal_hyperparams
        
    def run_experiment(self):
        """Run ablation experiment with enhanced metadata tracking."""
        print(f"\n🔬 Running component ablation: {self.variant_name}, seed {self.seed}")
        print(f"   Input shape: {self.input_shape}")
        
        try:
            # Create enhanced model wrapper
            model, hyperparams = self._create_enhanced_wrapper()
            
            ablation_config = self._get_ablation_config()
            print(f"Model created with ablation config:")
            print(f"  Attention: {ablation_config['use_attention_bottleneck']}")
            print(f"  Evidential: {ablation_config['use_evidential']}")
            print(f"  EVT: {ablation_config['use_evt']}")
            print(f"  Precursor: {ablation_config['use_precursor']}")
            
            # Train model with original wrapper (no monkey-patching)
            print(f"Training for 50 epochs with early stopping...")
            
            model_dir = model.train(
                X_train=self.X_train,
                y_train=self.y_train,
                epochs=50,
                batch_size=hyperparams["batch_size"],
                gamma_max=hyperparams["focal_gamma"],
                warmup_epochs=20,
                flare_class="M5",
                time_window="72",
                in_memory_dataset=True,
                track_emissions=False
            )
            
            print(f"✅ Training completed. Model saved to: {model_dir}")
            
            # Now evaluate on validation set and update metadata
            print(f"Evaluating on validation set...")
            y_pred_proba = model.predict_proba(self.X_val)
            y_pred = (y_pred_proba >= 0.5).astype(int).squeeze()
            
            # Debug: Show class distribution
            val_pos_count = np.sum(self.y_val)
            val_total = len(self.y_val)
            pred_pos_count = np.sum(y_pred)
            
            print(f"   • Validation set: {val_pos_count}/{val_total} positive ({val_pos_count/val_total:.1%})")
            print(f"   • Predictions: {pred_pos_count}/{val_total} positive ({pred_pos_count/val_total:.1%})")
            
            # Calculate comprehensive metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
            from sklearn.metrics import roc_auc_score, average_precision_score
            
            accuracy = accuracy_score(self.y_val, y_pred)
            precision = precision_score(self.y_val, y_pred, zero_division=0)
            recall = recall_score(self.y_val, y_pred, zero_division=0)
            f1 = f1_score(self.y_val, y_pred, zero_division=0)
            
            # Calculate TSS properly using confusion matrix
            cm = confusion_matrix(self.y_val, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            tss = sensitivity + specificity - 1
            
            try:
                auc_roc = roc_auc_score(self.y_val, y_pred_proba)
                auc_pr = average_precision_score(self.y_val, y_pred_proba)
            except:
                auc_roc = 0.0
                auc_pr = 0.0
            
            print(f"   • TSS: {tss:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc_roc:.4f}")
            
            # Update the saved metadata with enhanced metrics
            import json
            import os
            metadata_path = os.path.join(model_dir, "metadata.json")
            
            try:
                # Read existing metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Update with comprehensive validation metrics
                metadata["performance"].update({
                    "accuracy": accuracy,
                    "TSS": tss,
                    "precision": precision,
                    "recall": recall,
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "f1_score": f1,
                    "auc_roc": auc_roc,
                    "auc_pr": auc_pr,
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "true_negatives": int(tn),
                    "false_negatives": int(fn),
                    "positive_rate": float(np.mean(self.y_val)),
                    "prediction_rate": float(np.mean(y_pred))
                })
                
                # Add ablation-specific metadata
                metadata["hyperparameters"].update({
                    "ablation_variant": self.variant_name,
                    "ablation_seed": self.seed,
                    "learning_rate": hyperparams["learning_rate"],
                    "focal_gamma": hyperparams["focal_gamma"],
                    "batch_size": hyperparams["batch_size"]
                })
                
                metadata["ablation_metadata"] = {
                    "experiment_type": "component_ablation",
                    "variant": self.variant_name,
                    "seed": self.seed,
                    "ablation_config": ablation_config,
                    "optimal_hyperparams": hyperparams,
                    "description": ablation_config["description"]
                }
                
                # Update description
                metadata["description"] = f"EVEREST Ablation Study - {ablation_config['description']} (seed {self.seed})"
                
                # Save updated metadata
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"✅ Enhanced metadata saved to: {metadata_path}")
                
            except Exception as e:
                print(f"⚠️ Warning: Could not update metadata: {e}")
                # Continue anyway - the model is still saved
            
            results = {
                "experiment_type": "component_ablation",
                "variant": self.variant_name,
                "seed": self.seed,
                "input_shape": self.input_shape,
                "model_dir": model_dir,
                "ablation_config": ablation_config,
                "final_metrics": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "f1": f1,
                    "tss": tss,
                    "auc_roc": auc_roc,
                    "auc_pr": auc_pr,
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "true_negatives": int(tn),
                    "false_negatives": int(fn),
                    "positive_rate": float(np.mean(self.y_val)),
                    "prediction_rate": float(np.mean(y_pred))
                },
                "training_history": model.history
            }
            
            print(f"✅ Experiment completed successfully!")
            print(f"   • Variant: {self.variant_name} (seed {self.seed})")
            print(f"   • Accuracy: {accuracy:.4f}")
            print(f"   • TSS: {tss:.4f}")
            print(f"   • F1: {f1:.4f}")
            print(f"   • Sensitivity/Recall: {sensitivity:.4f}")
            print(f"   • Specificity: {specificity:.4f}")
            print(f"   • AUC-ROC: {auc_roc:.4f}")
            print(f"   • Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            print(f"   • Positive Rate: {float(np.mean(self.y_val)):.4f} | Prediction Rate: {float(np.mean(y_pred)):.4f}")
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
    print("🔬 EVEREST Component Ablation Study - Enhanced Metadata Tracking")
    print("   Component Ablations with Proper Variant/Seed Identification")
    print("   35 experiments: 7 variants × 5 seeds")
    print("=" * 80)


def validate_gpu():
    """Validate GPU configuration."""
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
            print(f"   ⚠️  GPU not available - checking if this is a local test...")
            import os
            if 'PBS_O_WORKDIR' not in os.environ and 'SLURM_JOB_ID' not in os.environ:
                print(f"   ℹ️  Local environment detected - GPU validation skipped for testing")
                print(f"   ⚠️  Note: This will fail on cluster without GPU!")
                return True
            else:
                print(f"   ❌ GPU not available on cluster - ablation requires GPU")
                return False
            
    except Exception as e:
        print(f"   ❌ GPU validation failed: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="EVEREST Component Ablation Study - Enhanced Metadata")
    
    parser.add_argument("--variant", 
                       choices=["full_model", "no_evidential", "no_evt", "mean_pool", 
                               "cross_entropy", "no_precursor", "fp32_training"],
                       required=True,
                       help="Component ablation variant to run")
    
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Validate GPU
    if not validate_gpu():
        print("\n❌ GPU validation failed!")
        return 1
    
    # Create and run ablation objective
    objective = AblationObjectiveWithMetadata(args.variant, args.seed)
    results = objective.run_experiment()
    
    if results:
        print(f"\n🎉 Ablation completed successfully!")
        print(f"📁 Results saved to: {results['model_dir']}")
        print(f"🏷️  Metadata includes: variant={args.variant}, seed={args.seed}")
        return 0
    else:
        print(f"\n❌ Ablation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 