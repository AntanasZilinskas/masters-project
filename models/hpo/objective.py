"""
Objective Function for EVEREST Hyperparameter Optimization

This module defines the objective function that Optuna will optimize,
integrating with the RETPlusWrapper and computing TSS as the primary metric.
"""

import os
import sys
import time
import traceback
from typing import Dict, Any, Tuple, Optional

import numpy as np
import optuna
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Add the models directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from solarknowledge_ret_plus import RETPlusWrapper
from utils import get_training_data, get_testing_data
from .config import (
    FIXED_ARCHITECTURE, 
    LOSS_WEIGHTS_CONFIG, 
    PERFORMANCE_THRESHOLDS,
    PRIMARY_METRIC,
    REPRODUCIBILITY_CONFIG
)


class HPOObjective:
    """
    Optuna objective function for EVEREST hyperparameter optimization.
    
    This class encapsulates the training and evaluation logic for a single
    trial, computing TSS (True Skill Statistic) as the primary metric.
    """
    
    def __init__(self, flare_class: str, time_window: str, use_validation: bool = True):
        """
        Initialize the objective function.
        
        Args:
            flare_class: Target flare class ("C", "M", "M5")
            time_window: Prediction window ("24", "48", "72")  
            use_validation: Whether to use separate validation data
        """
        self.flare_class = flare_class
        self.time_window = time_window
        self.use_validation = use_validation
        
        # Load data once to avoid repeated I/O
        self._load_data()
        
        # Set up reproducibility
        self._setup_reproducibility()
        
    def _setup_reproducibility(self) -> None:
        """Set up reproducible training environment."""
        seed = REPRODUCIBILITY_CONFIG["random_seed"]
        
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        # Set deterministic behavior (may impact performance)
        if REPRODUCIBILITY_CONFIG["torch_deterministic"]:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    def _load_data(self) -> None:
        """Load and cache training/validation data."""
        try:
            print(f"Loading data for {self.flare_class}-class, {self.time_window}h window...")
            
            # Load training data
            self.X_train, self.y_train = get_training_data(self.time_window, self.flare_class)
            
            if self.X_train is None or self.y_train is None:
                raise ValueError(f"Training data not found for {self.flare_class}/{self.time_window}h")
                
            # Load validation data (use testing data for validation during HPO)
            if self.use_validation:
                self.X_val, self.y_val = get_testing_data(self.time_window, self.flare_class)
                if self.X_val is None or self.y_val is None:
                    print("Warning: Validation data not found, using training data split")
                    self.use_validation = False
                    
            # Convert to numpy arrays
            self.X_train = np.array(self.X_train)
            self.y_train = np.array(self.y_train)
            
            if self.use_validation:
                self.X_val = np.array(self.X_val)
                self.y_val = np.array(self.y_val)
            else:
                # Simple train/val split if no separate validation data
                split_idx = int(0.8 * len(self.X_train))
                self.X_val = self.X_train[split_idx:]
                self.y_val = self.y_train[split_idx:]
                self.X_train = self.X_train[:split_idx]
                self.y_train = self.y_train[:split_idx]
                
            print(f"Data loaded successfully:")
            print(f"  Training: {self.X_train.shape[0]} samples")
            print(f"  Validation: {self.X_val.shape[0]} samples")
            print(f"  Positive rate (train): {self.y_train.mean():.3f}")
            print(f"  Positive rate (val): {self.y_val.mean():.3f}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
            
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for this trial."""
        return {
            "embed_dim": trial.suggest_categorical("embed_dim", [64, 128, 192, 256]),
            "num_blocks": trial.suggest_categorical("num_blocks", [4, 6, 8]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.40),
            "focal_gamma": trial.suggest_float("focal_gamma", 1.0, 4.0),
            "learning_rate": trial.suggest_float("learning_rate", 2e-4, 8e-4, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 768, 1024])
        }
        
    def _create_model(self, hyperparams: Dict[str, Any]) -> RETPlusWrapper:
        """Create model with suggested hyperparameters."""
        
        # Merge hyperparameters with fixed architecture
        model_config = {
            **FIXED_ARCHITECTURE,
            "embed_dim": hyperparams["embed_dim"],
            "num_heads": FIXED_ARCHITECTURE["num_heads"],
            "ff_dim": FIXED_ARCHITECTURE["ff_dim"], 
            "num_blocks": hyperparams["num_blocks"],
            "dropout": hyperparams["dropout"]
        }
        
        # Create wrapper (this will create the underlying model)
        wrapper = RETPlusWrapper(
            input_shape=model_config["input_shape"],
            use_attention_bottleneck=model_config["use_attention_bottleneck"],
            use_evidential=model_config["use_evidential"],
            use_evt=model_config["use_evt"],
            use_precursor=model_config["use_precursor"]
        )
        
        # Update optimizer with suggested learning rate
        wrapper.optimizer = torch.optim.AdamW(
            wrapper.model.parameters(),
            lr=hyperparams["learning_rate"],
            weight_decay=1e-4,
            fused=True
        )
        
        return wrapper
        
    def _train_and_evaluate(
        self, 
        model: RETPlusWrapper, 
        hyperparams: Dict[str, Any],
        epochs: int,
        trial: optuna.Trial
    ) -> Dict[str, float]:
        """Train model and evaluate on validation set."""
        
        # Custom training loop for intermediate pruning
        batch_size = hyperparams["batch_size"]
        focal_gamma = hyperparams["focal_gamma"]
        
        # Get device from model
        device = next(model.model.parameters()).device
        
        # Create data loaders
        from torch.utils.data import DataLoader, TensorDataset
        
        train_dataset = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.float32)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            pin_memory=(device.type == "cuda"), num_workers=2
        )
        
        val_dataset = TensorDataset(
            torch.tensor(self.X_val, dtype=torch.float32),
            torch.tensor(self.y_val, dtype=torch.float32)
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            pin_memory=(device.type == "cuda"), num_workers=2
        )
        
        best_tss = -float('inf')
        
        for epoch in range(epochs):
            # Training
            model.model.train()
            epoch_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                
                model.optimizer.zero_grad()
                
                # Dynamic loss weights based on epoch
                if epoch < 20:
                    weights = LOSS_WEIGHTS_CONFIG["phase_1"]
                elif epoch < 40:
                    weights = LOSS_WEIGHTS_CONFIG["phase_2"] 
                else:
                    weights = LOSS_WEIGHTS_CONFIG["phase_3"]
                
                outputs = model.model(X_batch)
                
                # Import composite loss from the model file
                from solarknowledge_ret_plus import composite_loss
                loss = composite_loss(
                    y_batch, outputs, gamma=focal_gamma, weights=weights
                )
                
                if not torch.isnan(loss):
                    loss.backward()
                    model.optimizer.step()
                    epoch_loss += loss.item()
                    
            # Validation evaluation
            if epoch % 1 == 0:  # Evaluate every epoch for pruning
                metrics = self._evaluate_model(model, val_loader, device)
                tss = metrics["TSS"]
                
                # Report intermediate value for pruning
                trial.report(tss, epoch)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
                    
                best_tss = max(best_tss, tss)
                
        # Final evaluation
        final_metrics = self._evaluate_model(model, val_loader, device)
        return final_metrics
        
    def _evaluate_model(self, model: RETPlusWrapper, val_loader, device) -> Dict[str, float]:
        """Evaluate model on validation set."""
        model.model.eval()
        
        all_preds = []
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)  # Ensure targets are also on device
                
                outputs = model.model(X_batch)
                probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_preds.extend(preds.flatten())
                all_probs.extend(probs.flatten())
                all_targets.extend(y_batch.cpu().numpy().flatten())
                
        # Compute metrics
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # TSS (True Skill Statistic)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall  # same as recall
        tss = sensitivity + specificity - 1
        
        # Additional metrics
        try:
            from sklearn.metrics import roc_auc_score, brier_score_loss
            roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
            brier = brier_score_loss(y_true, y_prob)
        except:
            roc_auc = 0.5
            brier = 1.0
            
        return {
            "TSS": tss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "roc_auc": roc_auc,
            "brier_score": brier,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn
        }
        
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Main objective function called by Optuna.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            TSS score (to be maximized)
        """
        start_time = time.time()
        
        # Initialize default values for user attributes
        stage = "unknown"
        metrics = {
            "TSS": -1.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "roc_auc": 0.5,
            "brier_score": 1.0,
            "tp": 0, "tn": 0, "fp": 0, "fn": 0
        }
        
        try:
            # Get hyperparameters for this trial
            hyperparams = self._suggest_hyperparameters(trial)
            
            # Determine epochs based on trial number (3-stage protocol)
            trial_number = trial.number
            if trial_number < 120:  # Exploration stage
                epochs = 20
                stage = "exploration"
            elif trial_number < 160:  # Refinement stage  
                epochs = 60
                stage = "refinement"
            else:  # Confirmation stage
                epochs = 120
                stage = "confirmation"
                
            print(f"\n🔍 Trial {trial_number} ({stage} stage)")
            print(f"Hyperparameters: {hyperparams}")
            print(f"Training for {epochs} epochs...")
            
            # Create and train model
            model = self._create_model(hyperparams)
            metrics = self._train_and_evaluate(model, hyperparams, epochs, trial)
            
            # Get primary metric (TSS)
            tss = metrics[PRIMARY_METRIC]
            
            elapsed = time.time() - start_time
            
            # Check performance thresholds but still save metrics
            if tss < PERFORMANCE_THRESHOLDS["min_tss"]:
                print(f"❌ TSS {tss:.4f} below threshold {PERFORMANCE_THRESHOLDS['min_tss']}")
                print(f"   Still saving metrics for analysis...")
            elif metrics["accuracy"] < PERFORMANCE_THRESHOLDS["min_accuracy"]:
                print(f"❌ Accuracy {metrics['accuracy']:.4f} below threshold {PERFORMANCE_THRESHOLDS['min_accuracy']}")
                print(f"   Still saving metrics for analysis...")
            else:
                print(f"✅ Trial completed in {elapsed:.1f}s")
                print(f"   TSS: {tss:.4f}, Accuracy: {metrics['accuracy']:.4f}")
                print(f"   Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
            
            return tss
            
        except optuna.TrialPruned:
            print(f"🛑 Trial {trial.number} pruned")
            elapsed = time.time() - start_time
            # Still save partial metrics for pruned trials
            trial.set_user_attr("pruned", True)
            raise
            
        except Exception as e:
            print(f"❌ Trial {trial.number} failed: {e}")
            print(traceback.format_exc())
            elapsed = time.time() - start_time
            # Save error information
            trial.set_user_attr("error", str(e))
            trial.set_user_attr("failed", True)
            return -1.0  # Return failed score for non-pruned failures
            
        finally:
            # Always save metrics and metadata, even for failed/pruned trials
            try:
                elapsed = time.time() - start_time
                
                # Store all metrics as user attributes
                for key, value in metrics.items():
                    if key != PRIMARY_METRIC:
                        # Ensure values are JSON serializable
                        if isinstance(value, (np.integer, np.floating)):
                            value = float(value)
                        trial.set_user_attr(key, value)
                        
                trial.set_user_attr("training_time", elapsed)
                trial.set_user_attr("stage", stage)
                trial.set_user_attr("flare_class", self.flare_class)
                trial.set_user_attr("time_window", self.time_window)
                
                # Add hyperparameters to user attributes for easier analysis
                for key, value in trial.params.items():
                    trial.set_user_attr(f"hp_{key}", value)
                    
            except Exception as attr_error:
                print(f"⚠️ Warning: Could not save user attributes: {attr_error}")
            
            # Do NOT return anything in finally block - it overrides try/except returns!


def create_objective(flare_class: str, time_window: str) -> HPOObjective:
    """Factory function to create objective for specific target."""
    return HPOObjective(flare_class, time_window)


if __name__ == "__main__":
    # Test the objective function
    from hpo.config import DEFAULT_TARGET
    
    print("Testing HPO objective function...")
    
    # Create test trial
    import optuna
    study = optuna.create_study(direction="maximize")
    
    # Create objective
    objective = create_objective(
        DEFAULT_TARGET["flare_class"], 
        DEFAULT_TARGET["time_window"]
    )
    
    # Run one trial
    try:
        study.optimize(objective, n_trials=1)
        print("✅ Objective function test completed successfully")
        print(f"Best TSS: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
    except Exception as e:
        print(f"❌ Objective function test failed: {e}")
        print(traceback.format_exc()) 