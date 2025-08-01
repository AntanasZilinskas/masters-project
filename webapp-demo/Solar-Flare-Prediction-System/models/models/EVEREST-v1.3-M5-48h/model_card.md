# EVEREST Model v1.3

## Overview
- **Created**: 2025-05-25 16:45
- **Type**: Solar flare prediction model
- **Target**: M5-class flares
- **Time Window**: 48 hours

## Description
EVEREST model trained on SHARP data with evidential and EVT losses.

## Performance Metrics
- **accuracy**: 0.9973
- **TSS**: 0.7624


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
- **Git Commit**: 6ae61ec49beb211dba5bc4841dce4a08d59e00c9
- **Git Branch**: pytorch-rewrite

## Usage
```python
from EVEREST_model import EVEREST

# Load the model
model = EVEREST()
model.load_model(
    input_shape=(10, 9),
    flare_class="M5",
    w_dir="EVEREST-v1.3-M5-48h"
)

# Make predictions
predictions = model.predict(X_test)
```
