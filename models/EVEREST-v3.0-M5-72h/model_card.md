# EVEREST Model v3.0

## Overview
- **Created**: 2025-05-28 07:49
- **Type**: Solar flare prediction model
- **Target**: M5-class flares
- **Time Window**: 72 hours

## Description
EVEREST model trained on SHARP data with evidential and EVT losses.

## Performance Metrics
- **accuracy**: 0.9966
- **TSS**: 0.5927
- **ROC_AUC**: 0.9952
- **Brier**: 0.0082
- **ECE**: 0.0584


## Training Details
- **Architecture**: EVEREST Transformer Model
- **Parameters**: 813,832
- **Precision**: torch.float32

## Hyperparameters
- **input_shape**: (10, 9)
- **embed_dim**: 128
- **num_heads**: 4
- **ff_dim**: 256
- **num_blocks**: 6
- **dropout**: 0.2

## Version Control
- **Git Commit**: 5ff1a61c7700d3630229b919aeaad50877aaf8a8
- **Git Branch**: pytorch-rewrite

## Usage
```python
from EVEREST_model import EVEREST

# Load the model
model = EVEREST()
model.load_model(
    input_shape=(10, 9),
    flare_class="M5",
    w_dir="EVEREST-v3.0-M5-72h"
)

# Make predictions
predictions = model.predict(X_test)
```
