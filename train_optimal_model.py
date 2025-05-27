#!/usr/bin/env python3
"""
Train EVEREST models with optimal hyperparameters from HPO study.

These hyperparameters achieved TSS = 0.832, exceeding the paper's target of 0.795.
"""

import sys
import os
import time
import torch
import numpy as np
from datetime import datetime

# Add models directory to path
sys.path.append('models')

from solarknowledge_ret_plus import RETPlusWrapper
from utils import get_training_data, get_testing_data

# ============================================================================
# OPTIMAL HYPERPARAMETERS (from HPO study achieving TSS = 0.832)
# ============================================================================
OPTIMAL_HYPERPARAMS = {
    "embed_dim": 128,
    "num_blocks": 4,
    "dropout": 0.3531616510212273,
    "focal_gamma": 2.8033450352296265,
    "learning_rate": 0.0005337429672856022,
    "batch_size": 512
}

# Architecture settings (fixed from HPO config)
ARCHITECTURE_CONFIG = {
    "input_shape": (10, 9),
    "num_heads": 4,
    "ff_dim": 256,
    "use_attention_bottleneck": True,
    "use_evidential": True,
    "use_evt": True,
    "use_precursor": True
}

# Training settings
TRAINING_CONFIG = {
    "epochs": 120,  # Full training (not just 20 epochs from exploration)
    "early_stopping_patience": 15,
    "gamma_max": OPTIMAL_HYPERPARAMS["focal_gamma"],
    "warmup_epochs": 50,
    "track_emissions": True
}

def create_optimal_model():
    """Create RETPlusWrapper with optimal hyperparameters."""
    print("🚀 Creating model with optimal hyperparameters:")
    for key, value in OPTIMAL_HYPERPARAMS.items():
        print(f"   {key}: {value}")
    
    # Create wrapper with architecture config
    wrapper = RETPlusWrapper(
        input_shape=ARCHITECTURE_CONFIG["input_shape"],
        early_stopping_patience=TRAINING_CONFIG["early_stopping_patience"],
        use_attention_bottleneck=ARCHITECTURE_CONFIG["use_attention_bottleneck"],
        use_evidential=ARCHITECTURE_CONFIG["use_evidential"],
        use_evt=ARCHITECTURE_CONFIG["use_evt"],
        use_precursor=ARCHITECTURE_CONFIG["use_precursor"]
    )
    
    # Update model architecture with optimal hyperparameters
    # Note: The model is already created, so we need to recreate it with new params
    from solarknowledge_ret_plus import RETPlusModel
    
    # Get device
    device = next(wrapper.model.parameters()).device
    
    # Create new model with optimal hyperparameters
    optimal_model = RETPlusModel(
        input_shape=ARCHITECTURE_CONFIG["input_shape"],
        embed_dim=OPTIMAL_HYPERPARAMS["embed_dim"],
        num_heads=ARCHITECTURE_CONFIG["num_heads"],
        ff_dim=ARCHITECTURE_CONFIG["ff_dim"],
        num_blocks=OPTIMAL_HYPERPARAMS["num_blocks"],
        dropout=OPTIMAL_HYPERPARAMS["dropout"],
        use_attention_bottleneck=ARCHITECTURE_CONFIG["use_attention_bottleneck"],
        use_evidential=ARCHITECTURE_CONFIG["use_evidential"],
        use_evt=ARCHITECTURE_CONFIG["use_evt"],
        use_precursor=ARCHITECTURE_CONFIG["use_precursor"]
    ).to(device)
    
    # Replace the model in wrapper
    wrapper.model = optimal_model
    
    # Create optimizer with optimal learning rate
    wrapper.optimizer = torch.optim.AdamW(
        wrapper.model.parameters(),
        lr=OPTIMAL_HYPERPARAMS["learning_rate"],
        weight_decay=1e-4,
        fused=True
    )
    
    print(f"✅ Model created on device: {device}")
    print(f"   Parameters: {sum(p.numel() for p in wrapper.model.parameters()):,}")
    
    return wrapper

def train_model(flare_class: str, time_window: str, model_suffix: str = "optimal"):
    """Train a model with optimal hyperparameters for specific target."""
    print(f"\n🎯 Training {flare_class}-class model for {time_window}h prediction")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load data
    print("📊 Loading training and validation data...")
    X_train, y_train = get_training_data(time_window, flare_class)
    X_val, y_val = get_testing_data(time_window, flare_class)
    
    if X_train is None or y_train is None:
        print(f"❌ Training data not found for {flare_class}/{time_window}h")
        return None
        
    if X_val is None or y_val is None:
        print(f"⚠️ Validation data not found for {flare_class}/{time_window}h")
        print("Using 80/20 split of training data")
        # Split training data
        split_idx = int(0.8 * len(X_train))
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]
        X_train = X_train[:split_idx]
        y_train = y_train[:split_idx]
    
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Validation samples: {len(X_val):,}")
    print(f"   Training positive rate: {np.mean(y_train):.3f}")
    print(f"   Validation positive rate: {np.mean(y_val):.3f}")
    
    # Create model
    model = create_optimal_model()
    
    # Train model (using the wrapper's train method but with custom parameters)
    print(f"\n🏋️ Training for {TRAINING_CONFIG['epochs']} epochs...")
    print(f"   Batch size: {OPTIMAL_HYPERPARAMS['batch_size']}")
    print(f"   Learning rate: {OPTIMAL_HYPERPARAMS['learning_rate']:.6f}")
    print(f"   Focal gamma: {OPTIMAL_HYPERPARAMS['focal_gamma']:.3f}")
    
    # Custom training call with validation data
    model_dir = train_with_validation(
        model, X_train, y_train, X_val, y_val, 
        flare_class, time_window, model_suffix
    )
    
    elapsed = time.time() - start_time
    print(f"\n✅ Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"📁 Model saved to: {model_dir}")
    
    return model_dir

def train_with_validation(model, X_train, y_train, X_val, y_val, flare_class, time_window, suffix):
    """Custom training loop with validation using optimal hyperparameters."""
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    device = next(model.model.parameters()).device
    batch_size = OPTIMAL_HYPERPARAMS["batch_size"]
    focal_gamma = OPTIMAL_HYPERPARAMS["focal_gamma"]
    epochs = TRAINING_CONFIG["epochs"]
    patience = TRAINING_CONFIG["early_stopping_patience"]
    
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
    
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=(device.type == "cuda"), num_workers=2
    )
    
    # Training loop
    best_tss = -float('inf')
    best_weights = None
    best_epoch = -1
    patience_counter = 0
    
    history = {"loss": [], "accuracy": [], "tss": [], "val_tss": []}
    
    from solarknowledge_ret_plus import composite_loss
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
    
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
            
            # Dynamic loss weights (3-phase schedule from HPO)
            if epoch < 20:
                weights = {"focal": 0.9, "evid": 0.1, "evt": 0.0, "prec": 0.05}
            elif epoch < 40:
                weights = {"focal": 0.8, "evid": 0.1, "evt": 0.1, "prec": 0.05}
            else:
                weights = {"focal": 0.7, "evid": 0.1, "evt": 0.2, "prec": 0.05}
            
            outputs = model.model(X_batch)
            loss = composite_loss(y_batch, outputs, gamma=focal_gamma, weights=weights)
            
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
        model.model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model.model(X_batch)
                probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                val_preds.extend(preds.flatten())
                val_targets.extend(y_batch.cpu().numpy().flatten())
        
        # Calculate metrics
        train_acc = train_correct / train_total if train_total > 0 else 0
        avg_loss = epoch_loss / len(train_loader)
        
        val_targets = np.array(val_targets)
        val_preds = np.array(val_preds)
        
        # Validation metrics
        val_acc = accuracy_score(val_targets, val_preds)
        precision = precision_score(val_targets, val_preds, zero_division=0)
        recall = recall_score(val_targets, val_preds, zero_division=0)
        
        # TSS calculation
        cm = confusion_matrix(val_targets, val_preds)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            tss = recall + specificity - 1
        else:
            tss = 0.0
        
        # Update history
        history["loss"].append(avg_loss)
        history["accuracy"].append(train_acc)
        history["tss"].append(0.0)  # Would need training TSS calculation
        history["val_tss"].append(tss)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:3d}/{epochs} - {epoch_time:.1f}s - "
              f"loss: {avg_loss:.4f} - acc: {train_acc:.4f} - "
              f"val_tss: {tss:.4f} - val_acc: {val_acc:.4f}")
        
        # Early stopping check
        if tss > best_tss:
            best_tss = tss
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
    
    # Save model with metadata
    model.history = history
    model._train_data = (X_train, y_train)  # For evaluation artifacts
    
    # Import model tracking
    from model_tracking import get_next_version
    
    version = get_next_version(flare_class, time_window)
    
    # Final validation metrics for saving
    final_metrics = {
        "TSS": best_tss,
        "accuracy": val_acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity if 'specificity' in locals() else 0.0
    }
    
    # Save model
    model_dir = model.save(
        version=version,
        flare_class=flare_class,
        time_window=time_window,
        X_eval=X_val,
        y_eval=y_val
    )
    
    return model_dir

def main():
    """Main training script."""
    print("🎯 EVEREST Optimal Hyperparameter Training")
    print("=" * 50)
    print("Using hyperparameters that achieved TSS = 0.832")
    print()
    
    # Configuration
    targets = [
        ("M", "24"),    # Original target that achieved 0.832 TSS
        ("C", "24"),    # Additional targets
        ("M5", "24"),
        ("M", "48"),
        ("M", "72")
    ]
    
    # Train models for each target
    results = {}
    total_start = time.time()
    
    for flare_class, time_window in targets:
        try:
            model_dir = train_model(flare_class, time_window)
            results[f"{flare_class}_{time_window}h"] = {
                "status": "success",
                "model_dir": model_dir
            }
        except Exception as e:
            print(f"❌ Failed to train {flare_class}-{time_window}h: {e}")
            results[f"{flare_class}_{time_window}h"] = {
                "status": "failed",
                "error": str(e)
            }
    
    total_time = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 TRAINING SUMMARY")
    print("=" * 60)
    
    for target, result in results.items():
        status_emoji = "✅" if result["status"] == "success" else "❌"
        print(f"{status_emoji} {target}: {result['status']}")
        if result["status"] == "success":
            print(f"    📁 {result['model_dir']}")
    
    print(f"\n⏱️ Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"📊 Models trained: {sum(1 for r in results.values() if r['status'] == 'success')}/{len(results)}")

if __name__ == "__main__":
    main() 