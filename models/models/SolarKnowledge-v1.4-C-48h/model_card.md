# SolarKnowledge Model v1.4

## Overview
- **Created**: 2025-04-22 02:50
- **Type**: Solar flare prediction model
- **Target**: C-class flares
- **Time Window**: 48 hours

## Description
EVEREST SHARP‑only experiment

## Performance Metrics
- **final_loss**: 0.0531
- **final_val_loss**: 0.0645
- **final_prec**: 0.9884
- **final_val_prec**: 0.9860
- **final_rec**: 0.9888
- **final_val_rec**: 0.9890
- **final_tss**: 0.9658
- **final_val_tss**: 0.9606
- **val_best_tss**: 0.9623
- **val_best_thr**: 0.7500


## Training Details
- **Architecture**: SolarKnowledge Transformer Model
- **Parameters**: 877,890
- **Precision**: <Policy "float32">

## Hyperparameters
- **linear_attention**: True

## Version Control
- **Git Commit**: 10100a7e93ffb8f19e4b08b8df720338abf57d31
- **Git Branch**: main

## Usage
```python
from SolarKnowledge_model import SolarKnowledge

# Load the model
model = SolarKnowledge()
model.load_model(
    input_shape=(10, 9), 
    flare_class="C", 
    w_dir="SolarKnowledge-v1.4-C-48h"
)

# Make predictions
predictions = model.predict(X_test)
```
