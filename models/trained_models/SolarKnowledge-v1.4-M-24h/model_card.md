# SolarKnowledge Model v1.4

## Overview
- **Created**: 2025-04-21 16:24
- **Type**: Solar flare prediction model
- **Target**: M-class flares
- **Time Window**: 24 hours

## Description
EVEREST SHARP‑only experiment

## Performance Metrics
- **final_loss**: 0.0089
- **final_prec**: 0.9731
- **final_rec**: 0.5026
- **final_tss**: 0.6105


## Training Details
- **Architecture**: SolarKnowledge Transformer Model
- **Parameters**: 862,209
- **Precision**: <Policy "float32">

## Hyperparameters
- **focal_alpha**: 0.322509351357939
- **focal_gamma**: 1.5
- **linear_attention**: True

## Version Control
- **Git Commit**: 365275083f60611db2b375dff63694da9beb259d
- **Git Branch**: EVEREST

## Usage
```python
from SolarKnowledge_model import SolarKnowledge

# Load the model
model = SolarKnowledge()
model.load_model(
    input_shape=(10, 9), 
    flare_class="M", 
    w_dir="SolarKnowledge-v1.4-M-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
