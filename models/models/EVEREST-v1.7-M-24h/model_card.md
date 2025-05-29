# EVEREST Model v1.7

## Overview
- **Created**: 2025-05-26 23:44
- **Type**: Solar flare prediction model
- **Target**: M-class flares
- **Time Window**: 24 hours

## Description
EVEREST model trained on SHARP data with evidential and EVT losses.

## Performance Metrics
- **accuracy**: 0.9911
- **TSS**: 0.9116
- **ROC_AUC**: 0.9992
- **Brier**: 0.0064
- **ECE**: 0.0156


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
- **Git Commit**: b8368982a9925bee8d4a7891ad7d39f2c63ac2c9
- **Git Branch**: pytorch-rewrite

## Usage
```python
from EVEREST_model import EVEREST

# Load the model
model = EVEREST()
model.load_model(
    input_shape=(10, 9),
    flare_class="M",
    w_dir="EVEREST-v1.7-M-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
