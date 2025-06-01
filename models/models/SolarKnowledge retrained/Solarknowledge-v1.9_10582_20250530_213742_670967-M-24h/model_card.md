# EVEREST Model v1.9_10582_20250530_213742_670967

## Overview
- **Created**: 2025-05-31 07:06
- **Type**: Solar flare prediction model
- **Target**: M-class flares
- **Time Window**: 24 hours

## Description
Iteration on v1.9 model for M-class flares with 24h window

## Performance Metrics
- **final_training_accuracy**: 0.9791
- **final_training_loss**: 0.0232
- **epochs_trained**: 100
- **final_training_tss**: 0.9521


## Training Details
- **Architecture**: EVEREST Transformer Model
- **Parameters**: 1,999,746
- **Precision**: <Policy "mixed_float16">

## Hyperparameters
- **learning_rate**: 0.0001
- **batch_size**: 512
- **early_stopping_patience**: 5
- **epochs**: 100
- **num_transformer_blocks**: 6
- **embed_dim**: 128
- **num_heads**: 4
- **ff_dim**: 256
- **dropout_rate**: 0.2
- **focal_loss**: True
- **focal_loss_alpha**: 0.25
- **focal_loss_gamma**: 2.0
- **class_weights**: {0: 1.0, 1: 6.822131674887412}
- **previous_version**: 1.9

## Version Control
- **Git Commit**: 8def88e0377157404ee1776a46748eadf0a3d12b
- **Git Branch**: pytorch-rewrite

## Usage
```python
from EVEREST_model import EVEREST

# Load the model
model = EVEREST()
model.load_model(
    input_shape=(10, 9),
    flare_class="M",
    w_dir="EVEREST-v1.9_10582_20250530_213742_670967-M-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
