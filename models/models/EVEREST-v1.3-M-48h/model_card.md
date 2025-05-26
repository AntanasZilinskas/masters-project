# EVEREST Model v1.3

## Overview
- **Created**: 2025-05-18 16:11
- **Type**: Solar flare prediction model
- **Target**: M-class flares
- **Time Window**: 48 hours

## Description
EVEREST model trained on SHARP data with evidential and EVT losses.

## Performance Metrics
- **accuracy**: 0.9921
- **TSS**: 0.9081
- **ROC_AUC**: 0.9991
- **Brier**: 0.0062
- **ECE**: 0.0170


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
    flare_class="M",
    w_dir="EVEREST-v1.3-M-48h"
)

# Make predictions
predictions = model.predict(X_test)
```
