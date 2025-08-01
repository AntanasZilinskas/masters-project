# EVEREST Model v1.1

## Overview
- **Created**: 2025-05-15 02:23
- **Type**: Solar flare prediction model
- **Target**: M-class flares
- **Time Window**: 48 hours

## Description
EVEREST model trained on SHARP data with evidential and EVT losses.

## Performance Metrics
- **accuracy**: 0.9927
- **TSS**: 0.9193


## Training Details
- **Architecture**: EVEREST Transformer Model
- **Parameters**: 814,089
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
    flare_class="M",
    w_dir="EVEREST-v1.1-M-48h"
)

# Make predictions
predictions = model.predict(X_test)
```
