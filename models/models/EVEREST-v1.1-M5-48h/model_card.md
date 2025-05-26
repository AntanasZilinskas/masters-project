# EVEREST Model v1.1

## Overview
- **Created**: 2025-05-17 13:15
- **Type**: Solar flare prediction model
- **Target**: M5-class flares
- **Time Window**: 48 hours

## Description
EVEREST model trained on SHARP data with evidential and EVT losses.

## Performance Metrics
- **accuracy**: 0.9926
- **TSS**: 0.0148
- **ROC_AUC**: 0.9264
- **Brier**: 0.0066
- **ECE**: 0.0037


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
    w_dir="EVEREST-v1.1-M5-48h"
)

# Make predictions
predictions = model.predict(X_test)
```
