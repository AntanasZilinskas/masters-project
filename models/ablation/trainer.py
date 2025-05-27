"""
Ablation Study Trainer for EVEREST

This module implements the training logic for systematic ablation studies,
following the experimental protocol with 5 random seeds per variant.
"""

import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from solarknowledge_ret_plus import RETPlusWrapper, RETPlusModel
from utils import get_training_data, get_testing_data
from .config import (
    OPTIMAL_HYPERPARAMS, FIXED_ARCHITECTURE, PRIMARY_TARGET,
    TRAINING_CONFIG, ABLATION_VARIANTS, SEQUENCE_LENGTH_VARIANTS,
    OUTPUT_CONFIG, get_variant_config, get_sequence_config,
    get_experiment_name
)


class AblationTrainer:
    """
    Trainer for systematic ablation studies.
    
    Implements the experimental protocol:
    - Fixed hyperparameters from HPO study
    - 5 random seeds per variant
    - Early stopping after 10 epochs
    - Proper loss weight re-normalization
    """
    
    def __init__(self, variant_name: str, seed: int, sequence_variant: Optional[str] = None):
        """
        Initialize ablation trainer.
        
        Args:
            variant_name: Name of ablation variant (e.g., 'no_evidential')
            seed: Random seed for reproducibility
            sequence_variant: Optional sequence length variant (e.g., 'seq_15')
        """
        self.variant_name = variant_name
        self.seed = seed
        self.sequence_variant = sequence_variant
        
        # Get variant configuration
        if variant_name not in ABLATION_VARIANTS:
            raise ValueError(f"Unknown variant: {variant_name}")
        
        self.variant_config = get_variant_config(variant_name)
        
        # Handle sequence length variants
        if sequence_variant:
            seq_config = get_sequence_config(sequence_variant)
            self.input_shape = seq_config["input_shape"]
        else:
            self.input_shape = FIXED_ARCHITECTURE["input_shape"]
        
        # Set up reproducibility
        self._setup_reproducibility()
        
        # Create experiment name and directories
        self.experiment_name = get_experiment_name(variant_name, seed, sequence_variant)
        self._setup_directories()
        
        print(f"🔬 Initialized ablation trainer: {self.experiment_name}")
        print(f"   Variant: {ABLATION_VARIANTS[variant_name]['name']}")
        if sequence_variant:
            print(f"   Sequence: {SEQUENCE_LENGTH_VARIANTS[sequence_variant]['name']}")
        print(f"   Seed: {seed}")
        print(f"   Input shape: {self.input_shape}")
    
    def _setup_reproducibility(self):
        """Set up reproducible training environment."""
        # Set random seeds
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        
        # Set deterministic behavior for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variable for additional reproducibility
        os.environ['PYTHONHASHSEED'] = str(self.seed)
    
    def _setup_directories(self):
        """Create experiment-specific directories."""
        self.experiment_dir = os.path.join(OUTPUT_CONFIG["results_dir"], self.experiment_name)
        self.model_dir = os.path.join(OUTPUT_CONFIG["models_dir"], self.experiment_name)
        self.log_dir = os.path.join(OUTPUT_CONFIG["logs_dir"], self.experiment_name)
        
        for directory in [self.experiment_dir, self.model_dir, self.log_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def create_model(self) -> RETPlusWrapper:
        """Create model with ablation-specific configuration."""
        print(f"🚀 Creating {self.variant_name} model...")
        
        # Create wrapper with ablation config
        wrapper = RETPlusWrapper(
            input_shape=self.input_shape,
            early_stopping_patience=FIXED_ARCHITECTURE["early_stopping_patience"],
            use_attention_bottleneck=self.variant_config["use_attention_bottleneck"],
            use_evidential=self.variant_config["use_evidential"],
            use_evt=self.variant_config["use_evt"],
            use_precursor=self.variant_config["use_precursor"],
            loss_weights=self.variant_config["loss_weights"]
        )
        
        # Get device
        device = next(wrapper.model.parameters()).device
        
        # Create model with optimal hyperparameters + ablation config
        ablation_model = RETPlusModel(
            input_shape=self.input_shape,
            embed_dim=OPTIMAL_HYPERPARAMS["embed_dim"],
            num_heads=FIXED_ARCHITECTURE["num_heads"],
            ff_dim=FIXED_ARCHITECTURE["ff_dim"],
            num_blocks=OPTIMAL_HYPERPARAMS["num_blocks"],
            dropout=OPTIMAL_HYPERPARAMS["dropout"],
            use_attention_bottleneck=self.variant_config["use_attention_bottleneck"],
            use_evidential=self.variant_config["use_evidential"],
            use_evt=self.variant_config["use_evt"],
            use_precursor=self.variant_config["use_precursor"]
        ).to(device)
        
        # Replace model in wrapper
        wrapper.model = ablation_model
        
        # Create optimizer with optimal learning rate
        wrapper.optimizer = torch.optim.AdamW(
            wrapper.model.parameters(),
            lr=OPTIMAL_HYPERPARAMS["learning_rate"],
            weight_decay=1e-4,
            fused=True
        )
        
        print(f"✅ Model created on device: {device}")
        print(f"   Parameters: {sum(p.numel() for p in wrapper.model.parameters()):,}")
        print(f"   Components: Evid={self.variant_config['use_evidential']}, "
              f"EVT={self.variant_config['use_evt']}, "
              f"Attn={self.variant_config['use_attention_bottleneck']}, "
              f"Prec={self.variant_config['use_precursor']}")
        
        return wrapper
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load training and validation data."""
        print("📊 Loading data...")
        
        flare_class = PRIMARY_TARGET["flare_class"]
        time_window = PRIMARY_TARGET["time_window"]
        
        # Load training and testing data
        X_train, y_train = get_training_data(time_window, flare_class)
        X_test, y_test = get_testing_data(time_window, flare_class)
        
        if X_train is None or y_train is None:
            raise ValueError(f"Training data not found for {flare_class}/{time_window}h")
        
        if X_test is None or y_test is None:
            raise ValueError(f"Testing data not found for {flare_class}/{time_window}h")
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # Handle sequence length variants by truncating/padding
        if self.input_shape[0] != 10:  # Default is 10 timesteps
            X_train = self._adjust_sequence_length(X_train, self.input_shape[0])
            X_test = self._adjust_sequence_length(X_test, self.input_shape[0])
        
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Testing samples: {len(X_test):,}")
        print(f"   Training positive rate: {np.mean(y_train):.4f}")
        print(f"   Testing positive rate: {np.mean(y_test):.4f}")
        print(f"   Input shape: {X_train.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def _adjust_sequence_length(self, X: np.ndarray, target_length: int) -> np.ndarray:
        """Adjust sequence length by truncating or padding."""
        current_length = X.shape[1]
        
        if target_length == current_length:
            return X
        elif target_length < current_length:
            # Truncate: take the most recent timesteps
            return X[:, -target_length:, :]
        else:
            # Pad: repeat the first timestep
            pad_length = target_length - current_length
            first_timestep = X[:, :1, :]  # Shape: (N, 1, F)
            padding = np.repeat(first_timestep, pad_length, axis=1)
            return np.concatenate([padding, X], axis=1)
    
    def _apply_ablation_to_weights(self, phase_weights: Dict[str, float]) -> Dict[str, float]:
        """Apply ablation-specific modifications to phase weights."""
        # Start with phase weights (these sum to 1.05 intentionally)
        ablation_weights = phase_weights.copy()
        
        # Apply ablation modifications (simply zero out disabled components)
        if not self.variant_config["use_evidential"]:
            ablation_weights["evid"] = 0.0
        
        if not self.variant_config["use_evt"]:
            ablation_weights["evt"] = 0.0
            
        if not self.variant_config["use_precursor"]:
            ablation_weights["prec"] = 0.0
        
        # Do NOT re-normalize - keep the same total weight as original training
        return ablation_weights
    
    def train(self, batch_size_override: Optional[int] = None, memory_efficient: bool = False) -> Dict[str, Any]:
        """Train the ablation model and return results."""
        start_time = time.time()
        
        print(f"\n🏋️ Training {self.experiment_name}")
        print("=" * 60)
        
        # Load data
        X_train, y_train, X_test, y_test = self.load_data()
        
        # Create model
        model = self.create_model()
        
        # Training configuration with optional overrides
        epochs = TRAINING_CONFIG["epochs"]
        batch_size = batch_size_override or OPTIMAL_HYPERPARAMS["batch_size"]
        
        # Apply memory efficient settings
        if memory_efficient:
            batch_size = min(batch_size, 512)  # Reduce batch size for memory sharing
            print("🧠 Memory efficient mode enabled")
        
        focal_gamma = self.variant_config["focal_gamma"]
        use_amp = self.variant_config["use_amp"]
        
        print(f"\n🎯 Training configuration:")
        print(f"   Epochs: {epochs} (early stop: {FIXED_ARCHITECTURE['early_stopping_patience']})")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {OPTIMAL_HYPERPARAMS['learning_rate']:.6f}")
        print(f"   Focal gamma: {focal_gamma}")
        print(f"   Mixed precision: {use_amp}")
        print(f"   Memory efficient: {memory_efficient}")
        print(f"   Loss weights: {self.variant_config['loss_weights']}")
        
        # Custom training loop for ablation studies
        results = self._train_with_validation(
            model, X_train, y_train, X_test, y_test,
            epochs, batch_size, focal_gamma, use_amp
        )
        
        # Save results
        self._save_results(results)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"📁 Results saved to: {self.experiment_dir}")
        
        return results
    
    def _train_with_validation(
        self, 
        model: RETPlusWrapper,
        X_train: np.ndarray,
        y_train: np.ndarray, 
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int,
        batch_size: int,
        focal_gamma: float,
        use_amp: bool
    ) -> Dict[str, Any]:
        """Custom training loop with validation."""
        
        device = next(model.model.parameters()).device
        
        # Create data loaders
        from torch.utils.data import DataLoader, TensorDataset
        
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            pin_memory=(device.type == "cuda"), num_workers=2
        )
        
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            pin_memory=(device.type == "cuda"), num_workers=2
        )
        
        # Training state
        best_tss = -float('inf')
        best_weights = None
        best_epoch = -1
        patience_counter = 0
        patience = FIXED_ARCHITECTURE["early_stopping_patience"]
        
        # History tracking
        history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "test_tss": [],
            "test_acc": [],
            "test_precision": [],
            "test_recall": [],
            "test_specificity": [],
            "test_f1": [],
            "test_brier": [],
            "test_roc_auc": []
        }
        
        # Import required functions
        from solarknowledge_ret_plus import composite_loss
        from sklearn.metrics import (
            confusion_matrix, accuracy_score, precision_score, 
            recall_score, f1_score, brier_score_loss, roc_auc_score
        )
        
        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        print(f"\n📈 Starting training loop...")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training phase
            model.model.train()
            epoch_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                
                model.optimizer.zero_grad()
                
                # Dynamic 3-phase weight schedule (matching main training)
                if epoch < 20:
                    phase_weights = {"focal": 0.9, "evid": 0.1, "evt": 0.0, "prec": 0.05}
                elif epoch < 40:
                    phase_weights = {"focal": 0.8, "evid": 0.1, "evt": 0.1, "prec": 0.05}
                else:
                    phase_weights = {"focal": 0.7, "evid": 0.1, "evt": 0.2, "prec": 0.05}
                
                # Apply ablation-specific modifications to phase weights
                ablation_weights = self._apply_ablation_to_weights(phase_weights)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model.model(X_batch)
                        loss = composite_loss(
                            y_batch, outputs, 
                            gamma=focal_gamma, 
                            weights=ablation_weights
                        )
                    
                    if not torch.isnan(loss):
                        scaler.scale(loss).backward()
                        scaler.step(model.optimizer)
                        scaler.update()
                        epoch_loss += loss.item()
                else:
                    outputs = model.model(X_batch)
                    loss = composite_loss(
                        y_batch, outputs,
                        gamma=focal_gamma,
                        weights=ablation_weights
                    )
                    
                    if not torch.isnan(loss):
                        loss.backward()
                        model.optimizer.step()
                        epoch_loss += loss.item()
                
                # Training accuracy
                with torch.no_grad():
                    probs = torch.sigmoid(outputs["logits"])
                    preds = (probs > 0.5).int()
                    train_correct += (preds.flatten() == y_batch.int().flatten()).sum().item()
                    train_total += y_batch.size(0)
            
            # Validation phase
            test_metrics = self._evaluate_model(model, test_loader, device)
            
            # Calculate training metrics
            train_acc = train_correct / train_total if train_total > 0 else 0
            avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
            
            # Update history
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(avg_loss)
            history["train_acc"].append(train_acc)
            for key, value in test_metrics.items():
                if f"test_{key}" in history:
                    history[f"test_{key}"].append(value)
            
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch+1:3d}/{epochs} - {epoch_time:.1f}s - "
                  f"loss: {avg_loss:.4f} - acc: {train_acc:.4f} - "
                  f"test_tss: {test_metrics['tss']:.4f} - "
                  f"test_acc: {test_metrics['accuracy']:.4f}")
            
            # Early stopping check
            current_tss = test_metrics['tss']
            if current_tss > best_tss:
                best_tss = current_tss
                best_weights = model.model.state_dict().copy()
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best TSS: {best_tss:.4f} at epoch {best_epoch+1}")
                break
        
        # Restore best weights
        if best_weights is not None:
            model.model.load_state_dict(best_weights)
        
        # Final evaluation
        final_metrics = self._evaluate_model(model, test_loader, device)
        
        # Measure inference latency
        latency_ms = self._measure_latency(model, X_test[:32], device)
        final_metrics["latency_ms"] = latency_ms
        
        return {
            "experiment_name": self.experiment_name,
            "variant_name": self.variant_name,
            "seed": self.seed,
            "sequence_variant": self.sequence_variant,
            "best_epoch": best_epoch + 1,
            "total_epochs": epoch + 1,
            "final_metrics": final_metrics,
            "history": history,
            "config": {
                "variant_config": self.variant_config,
                "hyperparams": OPTIMAL_HYPERPARAMS,
                "input_shape": self.input_shape
            }
        }
    
    def _evaluate_model(self, model: RETPlusWrapper, test_loader, device) -> Dict[str, float]:
        """Evaluate model on test set."""
        model.model.eval()
        
        all_preds = []
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model.model(X_batch)
                probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_preds.extend(preds.flatten())
                all_probs.extend(probs.flatten())
                all_targets.extend(y_batch.cpu().numpy().flatten())
        
        # Compute comprehensive metrics
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # TSS calculation
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            tss = recall + specificity - 1
        else:
            specificity = 0.0
            tss = 0.0
        
        # Calibration and probabilistic metrics
        try:
            roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
            brier = brier_score_loss(y_true, y_prob)
            ece = self._calculate_ece(y_true, y_prob, n_bins=15)
        except:
            roc_auc = 0.5
            brier = 1.0
            ece = 1.0
        
        return {
            "tss": tss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "roc_auc": roc_auc,
            "brier": brier,
            "ece": ece
        }
    
    def _calculate_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
        """Calculate Expected Calibration Error (15-bin protocol)."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _measure_latency(self, model: RETPlusWrapper, X_sample: np.ndarray, device, n_runs: int = 1000) -> float:
        """Measure inference latency in milliseconds."""
        model.model.eval()
        
        # Prepare input
        X_tensor = torch.tensor(X_sample, dtype=torch.float32).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model.model(X_tensor)
        
        # Measure latency
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model.model(X_tensor)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        end_time = time.time()
        
        # Convert to milliseconds per sample
        total_time_ms = (end_time - start_time) * 1000
        latency_per_batch = total_time_ms / n_runs
        latency_per_sample = latency_per_batch / len(X_sample)
        
        return latency_per_sample
    
    def _save_results(self, results: Dict[str, Any]):
        """Save experiment results."""
        # Save main results as JSON
        results_file = os.path.join(self.experiment_dir, "results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save history as CSV
        history_df = pd.DataFrame(results["history"])
        history_file = os.path.join(self.experiment_dir, "training_history.csv")
        history_df.to_csv(history_file, index=False)
        
        # Save final metrics as CSV
        metrics_df = pd.DataFrame([results["final_metrics"]])
        metrics_df["experiment_name"] = self.experiment_name
        metrics_df["variant_name"] = self.variant_name
        metrics_df["seed"] = self.seed
        metrics_df["sequence_variant"] = self.sequence_variant
        
        metrics_file = os.path.join(self.experiment_dir, "final_metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        
        print(f"📊 Results saved:")
        print(f"   JSON: {results_file}")
        print(f"   History: {history_file}")
        print(f"   Metrics: {metrics_file}")


def train_ablation_variant(variant_name: str, seed: int, sequence_variant: Optional[str] = None, 
                          batch_size_override: Optional[int] = None, memory_efficient: bool = False) -> Dict[str, Any]:
    """
    Train a single ablation variant with specified seed.
    
    Args:
        variant_name: Name of ablation variant
        seed: Random seed
        sequence_variant: Optional sequence length variant
        batch_size_override: Override batch size for memory optimization
        memory_efficient: Enable memory efficient training
        
    Returns:
        Training results dictionary
    """
    trainer = AblationTrainer(variant_name, seed, sequence_variant)
    return trainer.train(batch_size_override=batch_size_override, memory_efficient=memory_efficient)


if __name__ == "__main__":
    # Test training a single variant
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ablation study variant")
    parser.add_argument("--variant", required=True, choices=list(ABLATION_VARIANTS.keys()),
                        help="Ablation variant to train")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--sequence", choices=list(SEQUENCE_LENGTH_VARIANTS.keys()),
                        help="Sequence length variant")
    parser.add_argument("--batch-size", type=int, help="Override batch size for memory optimization")
    parser.add_argument("--memory-efficient", action="store_true", 
                        help="Enable memory efficient training (reduces batch size)")
    
    args = parser.parse_args()
    
    print(f"🔬 Training ablation variant: {args.variant}")
    print(f"🎲 Random seed: {args.seed}")
    if args.sequence:
        print(f"📏 Sequence variant: {args.sequence}")
    if args.batch_size:
        print(f"📦 Batch size override: {args.batch_size}")
    if args.memory_efficient:
        print(f"🧠 Memory efficient mode: enabled")
    
    results = train_ablation_variant(
        args.variant, 
        args.seed, 
        args.sequence,
        batch_size_override=args.batch_size,
        memory_efficient=args.memory_efficient
    )
    
    print(f"\n🎯 Final TSS: {results['final_metrics']['tss']:.4f}")
    print(f"🎯 Final F1: {results['final_metrics']['f1']:.4f}")
    print(f"⚡ Latency: {results['final_metrics']['latency_ms']:.1f} ms") 