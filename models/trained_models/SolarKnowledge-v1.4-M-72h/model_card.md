# SolarKnowledge Model v1.4

## Overview
- **Created**: 2025-04-23 04:15
- **Type**: Solar flare prediction model
- **Target**: M-class flares
- **Time Window**: 72 hours

## Description
EVEREST SHARP‑only experiment

## Performance Metrics
- **final_loss**: 0.0350
- **final_val_loss**: 0.0333
- **final_prec**: 0.9572
- **final_val_prec**: 0.9457
- **final_rec**: 0.9657
- **final_val_rec**: 0.9867
- **final_tss**: 0.9604
- **final_val_tss**: 0.9800
- **val_best_tss**: 0.9852
- **val_best_thr**: 0.1000


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
    flare_class="M", 
    w_dir="SolarKnowledge-v1.4-M-72h"
)

# Make predictions
predictions = model.predict(X_test)
```
