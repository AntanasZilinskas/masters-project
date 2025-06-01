# EVEREST Model v2.1_10582_20250531_195748_395683

## Overview
- **Created**: 2025-06-01 04:57
- **Type**: Solar flare prediction model
- **Target**: M-class flares
- **Time Window**: 48 hours

## Description
Iteration on v2.1 model for M-class flares with 48h window

## Performance Metrics
- **final_training_accuracy**: 0.9824
- **final_training_loss**: 0.0208
- **epochs_trained**: 100
- **final_training_tss**: 0.9565


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
- **class_weights**: {0: 1.0, 1: 7.595571249027471}
- **previous_version**: 2.1

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
    w_dir="EVEREST-v2.1_10582_20250531_195748_395683-M-48h"
)

# Make predictions
predictions = model.predict(X_test)
```
