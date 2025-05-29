# EVEREST Model v2.6

## Overview
- **Created**: 2025-05-26 12:34
- **Type**: Solar flare prediction model
- **Target**: C-class flares
- **Time Window**: 24 hours

## Description
EVEREST model trained on SHARP data with evidential and EVT losses.

## Performance Metrics
- **accuracy**: 0.9796
- **TSS**: 0.9591
- **ROC_AUC**: 0.9996
- **Brier**: 0.0106
- **ECE**: 0.0307


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
- **Git Commit**: 85ab6a9faee005d9c302055fb01cf8014281ead6
- **Git Branch**: pytorch-rewrite

## Usage
```python
from EVEREST_model import EVEREST

# Load the model
model = EVEREST()
model.load_model(
    input_shape=(10, 9),
    flare_class="C",
    w_dir="EVEREST-v2.6-C-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
