# SolarKnowledge Model v1.5

## Overview
- **Created**: 2025-04-21 17:03
- **Type**: Solar flare prediction model
- **Target**: M-class flares
- **Time Window**: 24 hours

## Description
EVEREST SHARP‑only experiment

## Performance Metrics
- **final_loss**: 0.7242
- **final_val_loss**: 0.4420
- **final_prec**: 0.7954
- **final_val_prec**: 0.8193
- **final_rec**: 0.7954
- **final_val_rec**: 0.8193
- **final_tss**: 0.0000
- **final_val_tss**: 0.0000
- **val_best_tss**: 0.6413
- **val_best_thr**: 0.5000


## Training Details
- **Architecture**: SolarKnowledge Transformer Model
- **Parameters**: 862,338
- **Precision**: <Policy "float32">

## Hyperparameters
- **linear_attention**: True

## Version Control
- **Git Commit**: 365275083f60611db2b375dff63694da9beb259d
- **Git Branch**: EVEREST

## Usage
```python
from SolarKnowledge_model import SolarKnowledge

# Load the model
model = SolarKnowledge()
model.load_model(
    input_shape=(10, 9), 
    flare_class="M", 
    w_dir="SolarKnowledge-v1.5-M-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
