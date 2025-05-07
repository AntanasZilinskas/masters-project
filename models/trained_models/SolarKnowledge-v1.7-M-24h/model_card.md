# SolarKnowledge Model v1.7

## Overview
- **Created**: 2025-04-23 01:18
- **Type**: Solar flare prediction model
- **Target**: M-class flares
- **Time Window**: 24 hours

## Description
EVEREST SHARP‑only experiment

## Performance Metrics
- **final_loss**: 0.0469
- **final_val_loss**: 0.0581
- **final_prec**: 0.9470
- **final_val_prec**: 0.9168
- **final_rec**: 0.9533
- **final_val_rec**: 0.9671
- **final_tss**: 0.9470
- **final_val_tss**: 0.9566
- **val_best_tss**: 0.9614
- **val_best_thr**: 0.1500


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
    w_dir="SolarKnowledge-v1.7-M-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
