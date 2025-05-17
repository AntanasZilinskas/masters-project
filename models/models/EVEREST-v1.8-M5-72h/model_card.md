# EVEREST Model v1.8

## Overview
- **Created**: 2025-05-16 08:05
- **Type**: Solar flare prediction model
- **Target**: M5-class flares
- **Time Window**: 72 hours

## Description
EVEREST model trained on SHARP data with evidential and EVT losses.

## Performance Metrics
- **accuracy**: 0.9981
- **TSS**: 0.8331
- **ROC_AUC**: 0.9989
- **Brier**: 0.0016
- **ECE**: 0.0093


## Training Details
- **Architecture**: EVEREST Transformer Model
- **Parameters**: 813,961
- **Precision**: torch.float32

## Hyperparameters
- **input_shape**: (10, 9)
- **embed_dim**: 128
- **num_heads**: 4
- **ff_dim**: 256
- **num_blocks**: 6
- **dropout**: 0.2

## Version Control
- **Git Commit**: 33d665b3d53cd8cce1dbf3a37da7f430c1ebc996
- **Git Branch**: pytorch-rewrite

## Usage
```python
from EVEREST_model import EVEREST

# Load the model
model = EVEREST()
model.load_model(
    input_shape=(10, 9),
    flare_class="M5",
    w_dir="EVEREST-v1.8-M5-72h"
)

# Make predictions
predictions = model.predict(X_test)
```
