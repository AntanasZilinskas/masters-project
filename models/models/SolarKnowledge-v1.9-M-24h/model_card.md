# SolarKnowledge Model v1.9

## Overview
- **Created**: 2025-05-11 11:17
- **Type**: Solar flare prediction model
- **Target**: M-class flares
- **Time Window**: 24 hours

## Description
EVEREST model trained on SHARP data with evidential and EVT losses.

## Performance Metrics
- **accuracy**: 0.9732
- **TSS**: 0.9464


## Training Details
- **Architecture**: SolarKnowledge Transformer Model
- **Parameters**: 814,089
- **Precision**: torch.float32

## Hyperparameters
- **input_shape**: (10, 9)
- **embed_dim**: 128
- **num_heads**: 4
- **ff_dim**: 256
- **num_blocks**: 6
- **dropout**: 0.2

## Version Control
- **Git Commit**: e86d19d349bc94c6b192ac39f05695dcb597378b
- **Git Branch**: pytorch-rewrite

## Usage
```python
from SolarKnowledge_model import SolarKnowledge

# Load the model
model = SolarKnowledge()
model.load_model(
    input_shape=(10, 9),
    flare_class="M",
    w_dir="SolarKnowledge-v1.9-M-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
