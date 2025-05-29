# EVEREST Model v1.9

## Overview
- **Created**: 2025-05-26 06:20
- **Type**: Solar flare prediction model
- **Target**: C-class flares
- **Time Window**: 48 hours

## Description
EVEREST model trained on SHARP data with evidential and EVT losses.

## Performance Metrics
- **accuracy**: 0.9765
- **TSS**: 0.9530
- **ROC_AUC**: 0.9995
- **Brier**: 0.0117
- **ECE**: 0.0339


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
    w_dir="EVEREST-v1.9-C-48h"
)

# Make predictions
predictions = model.predict(X_test)
```
