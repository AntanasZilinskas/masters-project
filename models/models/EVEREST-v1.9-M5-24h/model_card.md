# EVEREST Model v1.9

## Overview
- **Created**: 2025-05-15 20:50
- **Type**: Solar flare prediction model
- **Target**: M5-class flares
- **Time Window**: 24 hours

## Description
EVEREST model trained on SHARP data with evidential and EVT losses.

## Performance Metrics
- **accuracy**: 0.9954
- **TSS**: 0.6107
- **ROC_AUC**: 0.9976
- **Brier**: 0.0038
- **ECE**: 0.0079


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
    w_dir="EVEREST-v1.9-M5-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
