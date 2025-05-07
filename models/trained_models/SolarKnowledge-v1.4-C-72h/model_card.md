# SolarKnowledge Model v1.4

## Overview
- **Created**: 2025-04-22 23:30
- **Type**: Solar flare prediction model
- **Target**: C-class flares
- **Time Window**: 72 hours

## Description
EVEREST SHARP‑only experiment

## Performance Metrics
- **final_loss**: 0.0497
- **final_val_loss**: 0.0576
- **final_prec**: 0.9886
- **final_val_prec**: 0.9896
- **final_rec**: 0.9884
- **final_val_rec**: 0.9868
- **final_tss**: 0.9670
- **final_val_tss**: 0.9674
- **val_best_tss**: 0.9687
- **val_best_thr**: 0.5500


## Training Details
- **Architecture**: SolarKnowledge Transformer Model
- **Parameters**: 877,890
- **Precision**: <Policy "float32">

## Hyperparameters
- **linear_attention**: True

## Version Control
- **Git Commit**: 343a92f0baadb63993df65f4db22d8e1eaff9f2b
- **Git Branch**: data-pipeline

## Usage
```python
from SolarKnowledge_model import SolarKnowledge

# Load the model
model = SolarKnowledge()
model.load_model(
    input_shape=(10, 9), 
    flare_class="C", 
    w_dir="SolarKnowledge-v1.4-C-72h"
)

# Make predictions
predictions = model.predict(X_test)
```
