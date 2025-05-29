# EVEREST Model v2.6

## Overview
- **Created**: 2025-05-28 03:37
- **Type**: Solar flare prediction model
- **Target**: M5-class flares
- **Time Window**: 72 hours

## Description
EVEREST model trained on SHARP data with evidential and EVT losses.

## Performance Metrics
- **accuracy**: 0.9982
- **TSS**: 0.8347
- **ROC_AUC**: 0.9976
- **Brier**: 0.0021
- **ECE**: 0.0076


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
- **Git Commit**: c4cc7d3f03f6d3d50f8ff313979c4df4b1b6c272
- **Git Branch**: pytorch-rewrite

## Usage
```python
from EVEREST_model import EVEREST

# Load the model
model = EVEREST()
model.load_model(
    input_shape=(10, 9),
    flare_class="M5",
    w_dir="EVEREST-v2.6-M5-72h"
)

# Make predictions
predictions = model.predict(X_test)
```
