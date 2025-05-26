# EVEREST Model v1.9

## Overview
- **Created**: 2025-05-17 13:03
- **Type**: Solar flare prediction model
- **Target**: C-class flares
- **Time Window**: 24 hours

## Description
EVEREST model trained on SHARP data with evidential and EVT losses.

## Performance Metrics
- **accuracy**: 0.6951
- **TSS**: 0.3914
- **ROC_AUC**: 0.7753
- **Brier**: 0.1921
- **ECE**: 0.0223


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
    flare_class="C",
    w_dir="EVEREST-v1.9-C-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
