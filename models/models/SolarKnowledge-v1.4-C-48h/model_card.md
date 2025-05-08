# SolarKnowledge Model v1.4

## Overview
- **Created**: 2025-05-08 05:49
- **Type**: Solar flare prediction model
- **Target**: C-class flares
- **Time Window**: 48 hours

## Description
Iteration on v1.3 model for C-class flares with 48h window

## Performance Metrics
- **final_training_accuracy**: 0.9699
- **final_training_loss**: 0.0144
- **epochs_trained**: 100
- **final_training_tss**: 0.9434


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
- **class_weights**: {0: 1.0, 1: 0.4513797199127144}
- **previous_version**: 1.3

## Version Control
- **Git Commit**: e4293571a4735754d9690f194ea7de005d3aef73
- **Git Branch**: pytorch-rewrite

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
