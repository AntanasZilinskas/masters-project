# EVEREST Model v2.0

## Overview
- **Created**: 2025-05-25 05:01
- **Type**: Solar flare prediction model
- **Target**: M-class flares
- **Time Window**: 48 hours

## Description
EVEREST model trained on SHARP data with evidential and EVT losses. Early stopping on validation set.

## Performance Metrics
- **accuracy**: 0.9716
- **TSS**: -0.0000
- **loss**: 0.0974
- **best_val_epoch**: 1


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
- **lr**: 0.0003

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
    flare_class="M",
    w_dir="EVEREST-v2.0-M-48h"
)

# Make predictions
predictions = model.predict(X_test)
```
