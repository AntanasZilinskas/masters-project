# SolarKnowledge Model v1.4

## Overview
- **Created**: 2025-05-14 05:24
- **Type**: Solar flare prediction model
- **Target**: M5-class flares
- **Time Window**: 72 hours

## Description
EVEREST model trained on SHARP data with evidential and EVT losses.

## Performance Metrics
- **accuracy**: 0.9929
- **TSS**: 0.0138


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
- **Git Commit**: 8a49ad761ef96d1150be3557588c5980a5a340ee
- **Git Branch**: pytorch-rewrite

## Usage
```python
from SolarKnowledge_model import SolarKnowledge

# Load the model
model = SolarKnowledge()
model.load_model(
    input_shape=(10, 9),
    flare_class="M5",
    w_dir="SolarKnowledge-v1.4-M5-72h"
)

# Make predictions
predictions = model.predict(X_test)
```
