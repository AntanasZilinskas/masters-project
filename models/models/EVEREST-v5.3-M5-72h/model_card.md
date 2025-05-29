# EVEREST Model v5.3

## Overview
- **Created**: 2025-05-29 03:37
- **Type**: Solar flare prediction model
- **Target**: M5-class flares
- **Time Window**: 72 hours

## Description
EVEREST model trained on SHARP data with evidential and EVT losses.

## Performance Metrics
- **accuracy**: 0.9957
- **TSS**: 0.4726
- **ROC_AUC**: 0.9963
- **Brier**: 0.0052
- **ECE**: 0.0230


## Training Details
- **Architecture**: EVEREST Transformer Model
- **Parameters**: 549,001
- **Precision**: torch.float32

## Hyperparameters
- **input_shape**: (10, 9)
- **embed_dim**: 128
- **num_heads**: 4
- **ff_dim**: 256
- **num_blocks**: 6
- **dropout**: 0.2

## Version Control
- **Git Commit**: a48102e20990c0e78201f09ca6101336f0761a5c
- **Git Branch**: HEAD

## Usage
```python
from EVEREST_model import EVEREST

# Load the model
model = EVEREST()
model.load_model(
    input_shape=(10, 9),
    flare_class="M5",
    w_dir="EVEREST-v5.3-M5-72h"
)

# Make predictions
predictions = model.predict(X_test)
```
