# SolarKnowledge Model v1.4

## Overview
- **Created**: 2025-04-20 23:48
- **Type**: Solar flare prediction model
- **Target**: M5-class flares
- **Time Window**: 24 hours

## Description
EVEREST SHARP‑only experiment

## Performance Metrics
- **final_training_tss**: 0.7972
- **final_training_loss**: 0.0009


## Training Details
- **Architecture**: SolarKnowledge Transformer Model
- **Parameters**: 813,186
- **Precision**: <Policy "mixed_float16">

## Hyperparameters
- **focal_alpha**: 0.25
- **focal_gamma**: 2.0
- **linear_attention**: True

## Version Control
- **Git Commit**: 04e4669f8eadfd6c98f248fc597dd4108f147647
- **Git Branch**: EVEREST

## Usage
```python
from SolarKnowledge_model import SolarKnowledge

# Load the model
model = SolarKnowledge()
model.load_model(
    input_shape=(10, 9), 
    flare_class="M5", 
    w_dir="SolarKnowledge-v1.4-M5-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
