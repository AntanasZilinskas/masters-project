# EVEREST Model v1.0

## Overview
- **Created**: 2025-05-07 17:52
- **Type**: Solar flare prediction model
- **Target**: TEST-class flares
- **Time Window**: 24 hours

## Description
Test model for new directory structure

## Performance Metrics
- **accuracy**: 0.8500
- **precision**: 0.7500
- **recall**: 0.8000
- **tss**: 0.7000
- **final_loss**: 0.7000
- **final_val_loss**: 0.6500


## Training Details
- **Architecture**: EVEREST Model
- **Parameters**: 895,271
- **Precision**: <Policy "float32">

## Hyperparameters
- **learning_rate**: 0.0001
- **batch_size**: 32
- **dropout**: 0.3
- **embed_dim**: 128
- **num_transformer_blocks**: 4
- **uses_evidential**: True
- **uses_evt**: True

## Version Control
- **Git Commit**: 792851c750a6fbeab4b2484fbd3a5a819d6037bc
- **Git Branch**: EVEREST

## Usage
```python
from everest_model import EVEREST

# Load the model
model = EVEREST()
model.load_model(
    input_shape=[100, 14], 
    flare_class="TEST", 
    w_dir="EVEREST-v1.0-TEST-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
