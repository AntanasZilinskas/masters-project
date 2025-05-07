# SolarKnowledge Model v1.0

## Overview
- **Created**: 2025-05-06 02:53
- **Type**: Solar flare prediction model
- **Target**: M5-class flares
- **Time Window**: 24 hours

## Description
EVEREST-X enhanced with evidential uncertainty and EVT

## Performance Metrics
- **final_loss**: 0.1097
- **final_val_loss**: 0.4011
- **val_best_tss**: 0.8342
- **val_best_thr**: 0.3000
- **val_temp**: 0.1579


## Training Details
- **Architecture**: SolarKnowledge Transformer Model
- **Parameters**: 887,111
- **Precision**: <Policy "float32">

## Hyperparameters
- **linear_attention**: True
- **uses_evidential**: True
- **uses_evt**: True
- **uses_diffusion**: True

## Version Control
- **Git Commit**: 9ffb18319fccc3d4a68ec2ae5b445911fe102865
- **Git Branch**: EVEREST

## Usage
```python
from SolarKnowledge_model import SolarKnowledge

# Load the model
model = SolarKnowledge()
model.load_model(
    input_shape=(10, 9), 
    flare_class="M5", 
    w_dir="SolarKnowledge-v1.0-M5-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
