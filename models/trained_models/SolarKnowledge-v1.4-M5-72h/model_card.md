# SolarKnowledge Model v1.4

## Overview
- **Created**: 2025-04-23 06:34
- **Type**: Solar flare prediction model
- **Target**: M5-class flares
- **Time Window**: 72 hours

## Description
EVEREST SHARP‑only experiment

## Performance Metrics
- **final_loss**: 0.0635
- **final_val_loss**: 0.0630
- **final_prec**: 0.8516
- **final_val_prec**: 0.8214
- **final_rec**: 0.9485
- **final_val_rec**: 0.9787
- **final_tss**: 0.9476
- **final_val_tss**: 0.9773
- **val_best_tss**: 0.9992
- **val_best_thr**: 0.7000


## Training Details
- **Architecture**: SolarKnowledge Transformer Model
- **Parameters**: 877,890
- **Precision**: <Policy "float32">

## Hyperparameters
- **linear_attention**: True

## Version Control
- **Git Commit**: ef8a9eab7c62c4aea6bd2ce6a66ed4c0b95fbe6e
- **Git Branch**: EVEREST

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
