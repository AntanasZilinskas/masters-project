# SolarKnowledge Model v1.4

## Overview
- **Created**: 2025-04-22 00:59
- **Type**: Solar flare prediction model
- **Target**: C-class flares
- **Time Window**: 24 hours
- **Model Family**: EVEREST (Extreme‑Value/Evidential Retentive Event Sequence Transformer)

## Description
This model represents an improved version of the SolarKnowledge architecture that incorporates a transformer-based approach optimized for solar flare prediction. The EVEREST model features linear attention mechanisms that reduce computational complexity from O(L²) to O(L), which enables processing longer sequences of SHARP parameters (up to 72 hours of data).

Key improvements include:
- Multi-scale CNN stem that captures features at different temporal resolutions
- Performer-based linear attention for efficient sequence processing
- Monte Carlo dropout for uncertainty quantification
- Class-balanced focal loss to handle the imbalanced nature of solar flare events

This model achieves an exceptional TSS of 0.97 for C-class flare prediction, making it highly reliable for operational forecasting.

## Performance Metrics
- **final_loss**: 0.0434
- **final_val_loss**: 0.0572
- **final_prec**: 0.9917
- **final_val_prec**: 0.9893
- **final_rec**: 0.9924
- **final_val_rec**: 0.9890
- **final_tss**: 0.9750
- **final_val_tss**: 0.9659
- **val_best_tss**: 0.9717
- **val_best_thr**: 0.9000


## Training Details
- **Architecture**: EVEREST Transformer Model
- **Parameters**: 877,890
- **Precision**: float32
- **Blocks**: 6
- **Embedding Dimension**: 128
- **Attention Heads**: 4
- **Feedforward Dimension**: 256

## Hyperparameters
- **linear_attention**: True
- **multi_scale_cnn**: True
- **focal_loss**: True
- **dropout_rate**: 0.2

## Version Control
- **Git Commit**: 26348bff70c036607c1e5b4fe274bb380eba9875
- **Git Branch**: EVEREST

## Usage
```python
from everest_model import EVEREST

# Load the model
model = EVEREST()
model.build_base_model(input_shape=(10, 9))
model.load_weights(flare_class="C", w_dir="models/SolarKnowledge-v1.4-C-24h")

# Make predictions (standard)
predictions = model.predict_proba(X_test)

# For uncertainty quantification (Monte Carlo dropout)
mean_preds, uncertainties = model.mc_predict(X_test, n_passes=30)
```
