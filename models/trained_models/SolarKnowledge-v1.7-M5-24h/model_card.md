# SolarKnowledge Model v1.7

## Overview
- **Created**: 2025-04-21 03:10
- **Type**: Solar flare prediction model
- **Target**: M5-class flares
- **Time Window**: 24 hours

## Description
EVEREST SHARP‑only experiment

## Performance Metrics
- **final_training_tss**: 0.1043
- **final_training_loss**: 0.0023


## Training Details
- **Architecture**: SolarKnowledge Transformer Model
- **Parameters**: 862,338
- **Precision**: <Policy "float32">

## Hyperparameters
- **focal_alpha**: 0.5981108790582874
- **focal_gamma**: 2.0
- **linear_attention**: True

## Version Control
- **Git Commit**: f43400e1f35b00b1a5b0bd830f36b5d80a0d3759
- **Git Branch**: EVEREST

## Usage
```python
from SolarKnowledge_model import SolarKnowledge

# Load the model
model = SolarKnowledge()
model.load_model(
    input_shape=(10, 9), 
    flare_class="M5", 
    w_dir="SolarKnowledge-v1.7-M5-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
