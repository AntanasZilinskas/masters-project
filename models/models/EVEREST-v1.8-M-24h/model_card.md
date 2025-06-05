# EVEREST Model v1.8

## Overview
- **Created**: 2025-05-27 01:29
- **Type**: Solar flare prediction model
- **Target**: M-class flares
- **Time Window**: 24 hours

## Description
EVEREST model trained on SHARP data with evidential and EVT losses.

## Performance Metrics
- **accuracy**: 0.9412
- **TSS**: 0.0105
- **ROC_AUC**: 0.8279
- **Brier**: 0.0491
- **ECE**: 0.0065


## Training Details
- **Architecture**: EVEREST Transformer Model
- **Parameters**: 814,090
- **Precision**: torch.float32

## Hyperparameters
- **input_shape**: (10, 9)
- **embed_dim**: 128
- **num_heads**: 4
- **ff_dim**: 256
- **num_blocks**: 6
- **dropout**: 0.2

## Version Control
- **Git Commit**: d64d6657bf3770a03f8799ef9ebbce2e37182493
- **Git Branch**: pytorch-rewrite

## Usage
```python
from EVEREST_model import EVEREST

# Load the model
model = EVEREST()
model.load_model(
    input_shape=(10, 9),
    flare_class="M",
    w_dir="EVEREST-v1.8-M-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
