# EVEREST Model v1.0

## Overview
- **Created**: 2025-05-14 06:59
- **Type**: Solar flare prediction model
- **Target**: M-class flares
- **Time Window**: 48 hours

## Description
EVEREST model trained on SHARP data with evidential and EVT losses.

## Performance Metrics
- **accuracy**: 0.9470
- **TSS**: 0.0059


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
- **Git Commit**: 8a49ad761ef96d1150be3557588c5980a5a340ee
- **Git Branch**: pytorch-rewrite

## Usage
```python
from EVEREST_model import EVEREST

# Load the model
model = EVEREST()
model.load_model(
    input_shape=(10, 9),
    flare_class="M",
    w_dir="EVEREST-v1.0-M-48h"
)

# Make predictions
predictions = model.predict(X_test)
```
