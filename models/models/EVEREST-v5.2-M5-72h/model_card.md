# EVEREST Model v5.2

## Overview
- **Created**: 2025-05-29 03:13
- **Type**: Solar flare prediction model
- **Target**: M5-class flares
- **Time Window**: 72 hours

## Description
EVEREST model trained on SHARP data with evidential and EVT losses.

## Performance Metrics
- **accuracy**: 0.9958
- **TSS**: 0.4879
- **ROC_AUC**: 0.9968
- **Brier**: 0.0047
- **ECE**: 0.0178


## Training Details
- **Architecture**: EVEREST Transformer Model
- **Parameters**: 548,614
- **Precision**: torch.float32

## Hyperparameters
- **input_shape**: (10, 9)
- **embed_dim**: 128
- **num_heads**: 4
- **ff_dim**: 256
- **num_blocks**: 6
- **dropout**: 0.2

## Version Control
- **Git Commit**: 0cbc56edf81be6250c0c43de7c244eb04ba1dfee
- **Git Branch**: pytorch-rewrite

## Usage
```python
from EVEREST_model import EVEREST

# Load the model
model = EVEREST()
model.load_model(
    input_shape=(10, 9),
    flare_class="M5",
    w_dir="EVEREST-v5.2-M5-72h"
)

# Make predictions
predictions = model.predict(X_test)
```
