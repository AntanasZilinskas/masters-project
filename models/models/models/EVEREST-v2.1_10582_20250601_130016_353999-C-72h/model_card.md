# EVEREST Model v2.1_10582_20250601_130016_353999

## Overview
- **Created**: 2025-06-01 21:20
- **Type**: Solar flare prediction model
- **Target**: C-class flares
- **Time Window**: 72 hours

## Description
Iteration on v2.1 model for C-class flares with 72h window

## Performance Metrics
- **final_training_accuracy**: 0.9332
- **final_training_loss**: 0.0307
- **epochs_trained**: 100
- **final_training_tss**: 0.8692


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
- **class_weights**: {0: 1.0, 1: 0.574999676295788}
- **previous_version**: 2.1

## Version Control
- **Git Commit**: 0e1b01bde32ff9fe938e7ee415bbc8ff30cc83a2
- **Git Branch**: pytorch-rewrite

## Usage
```python
from EVEREST_model import EVEREST

# Load the model
model = EVEREST()
model.load_model(
    input_shape=(10, 9),
    flare_class="C",
    w_dir="EVEREST-v2.1_10582_20250601_130016_353999-C-72h"
)

# Make predictions
predictions = model.predict(X_test)
```
