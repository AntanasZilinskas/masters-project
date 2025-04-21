# SolarKnowledge Model v1.6

## Overview
- **Created**: 2025-04-21 18:40
- **Type**: Solar flare prediction model
- **Target**: M-class flares
- **Time Window**: 24 hours

## Description
EVEREST SHARP‑only experiment

## Performance Metrics
- **final_loss**: 0.0736
- **final_val_loss**: 0.0766
- **final_prec**: 0.9347
- **final_val_prec**: 0.9266
- **final_rec**: 0.9034
- **final_val_rec**: 0.9232
- **final_tss**: 0.8959
- **final_val_tss**: 0.9145
- **val_best_tss**: 0.9542
- **val_best_thr**: 0.0500


## Training Details
- **Architecture**: SolarKnowledge Transformer Model
- **Parameters**: 877,890
- **Precision**: <Policy "float32">

## Hyperparameters
- **linear_attention**: True

## Version Control
- **Git Commit**: 0145a5e209affb4bb73f682188f2b0cd7d1528ea
- **Git Branch**: EVEREST

## Usage
```python
from SolarKnowledge_model import SolarKnowledge

# Load the model
model = SolarKnowledge()
model.load_model(
    input_shape=(10, 9), 
    flare_class="M", 
    w_dir="SolarKnowledge-v1.6-M-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
