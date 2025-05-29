# EVEREST Model v3.7

## Overview
- **Created**: 2025-05-28 18:34
- **Type**: Solar flare prediction model
- **Target**: M5-class flares
- **Time Window**: 72 hours

## Description
EVEREST model trained on SHARP data with evidential and EVT losses.

## Performance Metrics
- **accuracy**: 0.9966
- **TSS**: 0.6051
- **ROC_AUC**: 0.9960
- **Brier**: 0.0087
- **ECE**: 0.0705


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
- **Git Commit**: 67f282a9529899517317538d2bf6ac77a70ae0c1
- **Git Branch**: pytorch-rewrite

## Usage
```python
from EVEREST_model import EVEREST

# Load the model
model = EVEREST()
model.load_model(
    input_shape=(10, 9),
    flare_class="M5",
    w_dir="EVEREST-v3.7-M5-72h"
)

# Make predictions
predictions = model.predict(X_test)
```
