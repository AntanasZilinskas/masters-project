# SolarKnowledge Model v1.8

## Overview
- **Created**: 2025-04-21 19:11
- **Type**: Solar flare prediction model
- **Target**: M5-class flares
- **Time Window**: 24 hours
- **Model Family**: EVEREST (Extreme‑Value/Evidential Retentive Event Sequence Transformer)

## Description
This model represents the latest version of the EVEREST architecture optimized specifically for major (M5+) flare prediction. M5+ flares are extremely rare events, making their prediction particularly challenging due to severe class imbalance.

Key features of this model:
- Multi-scale CNN stem with 3/5/7 causal convolutions to capture temporal patterns at different scales
- Linear-attention transformer blocks (Performer-based) for efficient sequence processing
- Hybrid loss function that directly optimizes for the TSS metric (0.8*BCE + 0.2*TSS surrogate)
- Double-dropout regularization strategy with 70% application probability
- Group-wise train/validation split to prevent data leakage across active regions

The model achieves a validation TSS of 0.67, which represents state-of-the-art performance for M5+ flare prediction with a 24-hour forecast window.

## Performance Metrics
- **final_loss**: 0.1639
- **final_val_loss**: 0.1668
- **final_prec**: 0.7540
- **final_val_prec**: 0.6750
- **final_rec**: 0.6247
- **final_val_rec**: 0.6279
- **final_tss**: 0.6229
- **final_val_tss**: 0.6251
- **val_best_tss**: 0.6714
- **val_best_thr**: 0.5000


## Training Details
- **Architecture**: EVEREST Transformer Model
- **Parameters**: 877,890
- **Precision**: float32
- **Blocks**: 6
- **Embedding Dimension**: 128
- **Attention Heads**: 4
- **Feedforward Dimension**: 256
- **Double-Dropout**: Enabled (70% application probability)

## Hyperparameters
- **linear_attention**: True
- **multi_scale_cnn**: True
- **hybrid_loss**: True
- **dropout_rate**: 0.2
- **double_dropout**: True
- **double_dropout_rate**: 0.3

## Version Control
- **Git Commit**: 843091b146f58d38e0976c9ffc022a14f8427b6d
- **Git Branch**: EVEREST

## Usage
```python
from everest_model import EVEREST

# Load the model
model = EVEREST()
model.build_base_model(input_shape=(10, 9))
model.load_weights(flare_class="M5", w_dir="models/SolarKnowledge-v1.8-M5-24h")

# Make predictions (standard)
predictions = model.predict_proba(X_test)

# For uncertainty quantification (Monte Carlo dropout)
mean_preds, uncertainties = model.mc_predict(X_test, n_passes=30)

# Apply temperature scaling for calibrated probabilities
calibrated_preds = mean_preds / T  # where T is the temperature parameter
```
