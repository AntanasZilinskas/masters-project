# SolarKnowledge Model v1.4

## Overview
- **Created**: 2025-04-23 02:11
- **Type**: Solar flare prediction model
- **Target**: M-class flares
- **Time Window**: 48 hours

## Description
EVEREST SHARP‑only experiment

## Performance Metrics
- **final_loss**: 0.0568
- **final_val_loss**: 0.0627
- **final_prec**: 0.9471
- **final_val_prec**: 0.9398
- **final_rec**: 0.9335
- **final_val_rec**: 0.9230
- **final_tss**: 0.9274
- **final_val_tss**: 0.9158
- **val_best_tss**: 0.9545
- **val_best_thr**: 0.1000


## Training Details
- **Architecture**: SolarKnowledge Transformer Model
- **Parameters**: 877,890
- **Precision**: <Policy "float32">

## Hyperparameters
- **linear_attention**: True

## Version Control
- **Git Commit**: d5a86320c03a42dda89186b8ad38424d1bafdb81
- **Git Branch**: EVEREST

## Usage
```python
from SolarKnowledge_model import SolarKnowledge

# Load the model
model = SolarKnowledge()
model.load_model(
    input_shape=(10, 9), 
    flare_class="M", 
    w_dir="SolarKnowledge-v1.4-M-48h"
)

# Make predictions
predictions = model.predict(X_test)
```
