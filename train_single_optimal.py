#!/usr/bin/env python3
"""
Simple script to train a single EVEREST model with optimal hyperparameters.
"""

import sys
import os
import argparse

# Add models directory to path
sys.path.append('models')

from solarknowledge_ret_plus import RETPlusWrapper, RETPlusModel
from utils import get_training_data, get_testing_data

# Optimal hyperparameters from HPO study (TSS = 0.832)
OPTIMAL_PARAMS = {
    "embed_dim": 128,
    "num_blocks": 4,
    "dropout": 0.3531616510212273,
    "focal_gamma": 2.8033450352296265,
    "learning_rate": 0.0005337429672856022,
    "batch_size": 512
}

def train_optimal_model(flare_class="M", time_window="24", epochs=120):
    """Train a single model with optimal hyperparameters."""
    
    print("🎯 EVEREST Optimal Training")
    print(f"Target: {flare_class}-class flares, {time_window}h window")
    print("Optimal hyperparameters:")
    for k, v in OPTIMAL_PARAMS.items():
        print(f"  {k}: {v}")
    print()
    
    # Load data
    print("📊 Loading data...")
    X_train, y_train = get_training_data(time_window, flare_class)
    X_val, y_val = get_testing_data(time_window, flare_class)
    
    if X_train is None or y_train is None:
        print(f"❌ No training data found for {flare_class}/{time_window}h")
        return None
    
    print(f"Training samples: {len(X_train):,}")
    if X_val is not None and y_val is not None:
        print(f"Validation samples: {len(X_val):,}")
    else:
        print("⚠️ No separate validation data, will use training split")
    
    # Create model with optimal architecture
    print("\n🚀 Creating model...")
    
    # Create wrapper
    wrapper = RETPlusWrapper(
        input_shape=(10, 9),
        early_stopping_patience=15,
        use_attention_bottleneck=True,
        use_evidential=True,
        use_evt=True,
        use_precursor=True
    )
    
    # Get device
    device = next(wrapper.model.parameters()).device
    print(f"Device: {device}")
    
    # Create model with optimal hyperparameters
    optimal_model = RETPlusModel(
        input_shape=(10, 9),
        embed_dim=OPTIMAL_PARAMS["embed_dim"],
        num_heads=4,  # Fixed from architecture
        ff_dim=256,   # Fixed from architecture
        num_blocks=OPTIMAL_PARAMS["num_blocks"],
        dropout=OPTIMAL_PARAMS["dropout"],
        use_attention_bottleneck=True,
        use_evidential=True,
        use_evt=True,
        use_precursor=True
    ).to(device)
    
    # Replace model in wrapper
    wrapper.model = optimal_model
    
    # Create optimizer with optimal learning rate
    import torch
    wrapper.optimizer = torch.optim.AdamW(
        wrapper.model.parameters(),
        lr=OPTIMAL_PARAMS["learning_rate"],
        weight_decay=1e-4,
        fused=True
    )
    
    print(f"Parameters: {sum(p.numel() for p in wrapper.model.parameters()):,}")
    
    # Train model
    print(f"\n🏋️ Training for {epochs} epochs...")
    print(f"Batch size: {OPTIMAL_PARAMS['batch_size']}")
    print(f"Learning rate: {OPTIMAL_PARAMS['learning_rate']:.6f}")
    print(f"Focal gamma: {OPTIMAL_PARAMS['focal_gamma']:.3f}")
    
    # Use the wrapper's built-in training but override key parameters
    if X_val is not None and y_val is not None:
        # Custom training with validation
        model_dir = wrapper.train(
            X_train, y_train,
            epochs=epochs,
            batch_size=OPTIMAL_PARAMS["batch_size"],
            gamma_max=OPTIMAL_PARAMS["focal_gamma"],
            flare_class=flare_class,
            time_window=time_window,
            track_emissions=False  # Disable for speed
        )
    else:
        # Standard training without separate validation
        model_dir = wrapper.train(
            X_train, y_train,
            epochs=epochs,
            batch_size=OPTIMAL_PARAMS["batch_size"],
            gamma_max=OPTIMAL_PARAMS["focal_gamma"],
            flare_class=flare_class,
            time_window=time_window,
            track_emissions=False
        )
    
    print(f"\n✅ Training complete!")
    print(f"📁 Model saved to: {model_dir}")
    
    # Display final metrics
    if hasattr(wrapper, 'history'):
        final_tss = wrapper.history.get('tss', [0])[-1] if wrapper.history.get('tss') else 0
        final_acc = wrapper.history.get('accuracy', [0])[-1] if wrapper.history.get('accuracy') else 0
        print(f"🎯 Final TSS: {final_tss:.4f}")
        print(f"🎯 Final accuracy: {final_acc:.4f}")
    
    return model_dir

def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description="Train EVEREST with optimal hyperparameters")
    parser.add_argument("--flare_class", "-c", default="M", choices=["C", "M", "M5"],
                        help="Flare class to predict (default: M)")
    parser.add_argument("--time_window", "-t", default="24", choices=["24", "48", "72"],
                        help="Prediction time window in hours (default: 24)")
    parser.add_argument("--epochs", "-e", type=int, default=120,
                        help="Number of training epochs (default: 120)")
    
    args = parser.parse_args()
    
    print("🌟 Starting EVEREST training with optimal hyperparameters")
    print(f"   Configuration: {args.flare_class}-class, {args.time_window}h, {args.epochs} epochs")
    print(f"   Expected TSS: ~0.832 (based on HPO study)")
    print()
    
    model_dir = train_optimal_model(
        flare_class=args.flare_class,
        time_window=args.time_window,
        epochs=args.epochs
    )
    
    if model_dir:
        print(f"\n🎉 Success! Model ready for deployment.")
        print(f"📂 Location: {model_dir}")
    else:
        print(f"\n❌ Training failed.")

if __name__ == "__main__":
    main() 