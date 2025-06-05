# SolarKnowledge Model v1.0

## Overview
- **Created**: 2025-05-08 20:10
- **Type**: Solar flare prediction model
- **Target**: C-class flares
- **Time Window**: 24 hours

## Description
Initial model for C-class flares with 24h prediction window

## Performance Metrics
- **final_training_accuracy**: 0.9681
- **final_training_loss**: 0.0173
- **epochs_trained**: 100
- **final_training_tss**: 0.9407


## Training Details
- **Architecture**: SolarKnowledge Transformer Model
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
- **class_weights**: {0: 1.0, 1: 0.4421536237869089}

## Version Control
- **Git Commit**: 792419d333ff3f4ead8c44167ef94ab8eaccc0fc
- **Git Branch**: pytorch-rewrite

## Usage
```python
from SolarKnowledge_model import SolarKnowledge

# Load the model
model = SolarKnowledge()
model.load_model(
    input_shape=(10, 9),
    flare_class="C",
    w_dir="SolarKnowledge-v1.0-C-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
