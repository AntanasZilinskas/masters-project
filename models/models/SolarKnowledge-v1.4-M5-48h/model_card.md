# SolarKnowledge Model v1.4

## Overview
- **Created**: 2025-05-08 07:01
- **Type**: Solar flare prediction model
- **Target**: M5-class flares
- **Time Window**: 48 hours

## Description
Iteration on v1.3 model for M5-class flares with 48h window

## Performance Metrics
- **final_training_accuracy**: 0.9885
- **final_training_loss**: 0.0386
- **epochs_trained**: 46
- **final_training_tss**: 0.9789


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
- **class_weights**: {0: 1.0, 1: 150.6346153846154}
- **previous_version**: 1.3

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
    flare_class="M5",
    w_dir="SolarKnowledge-v1.4-M5-48h"
)

# Make predictions
predictions = model.predict(X_test)
```
