# EVEREST Model v2.0

## Overview
- **Created**: 2025-05-26 22:38
- **Type**: Solar flare prediction model
- **Target**: C-class flares
- **Time Window**: 72 hours

## Description
EVEREST model trained on SHARP data with evidential and EVT losses.

## Performance Metrics
- **accuracy**: 0.9749
- **TSS**: 0.9498
- **ROC_AUC**: 0.9995
- **Brier**: 0.0123
- **ECE**: 0.0365


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
- **Git Commit**: b8368982a9925bee8d4a7891ad7d39f2c63ac2c9
- **Git Branch**: pytorch-rewrite

## Usage
```python
from EVEREST_model import EVEREST

# Load the model
model = EVEREST()
model.load_model(
    input_shape=(10, 9),
    flare_class="C",
    w_dir="EVEREST-v2.0-C-72h"
)

# Make predictions
predictions = model.predict(X_test)
```
