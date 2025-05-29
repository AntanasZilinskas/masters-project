# EVEREST Model v2.5

## Overview
- **Created**: 2025-05-28 02:27
- **Type**: Solar flare prediction model
- **Target**: M5-class flares
- **Time Window**: 72 hours

## Description
EVEREST model trained on SHARP data with evidential and EVT losses.

## Performance Metrics
- **accuracy**: 0.9929
- **TSS**: 0.0042
- **ROC_AUC**: 0.9340
- **Brier**: 0.0062
- **ECE**: 0.0024


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
- **Git Commit**: def9268ae43c170a23711491eb66ed03dfe0117e
- **Git Branch**: pytorch-rewrite

## Usage
```python
from EVEREST_model import EVEREST

# Load the model
model = EVEREST()
model.load_model(
    input_shape=(10, 9),
    flare_class="M5",
    w_dir="EVEREST-v2.5-M5-72h"
)

# Make predictions
predictions = model.predict(X_test)
```
