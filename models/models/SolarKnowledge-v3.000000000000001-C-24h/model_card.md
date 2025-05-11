# SolarKnowledge Model v3.000000000000001

## Overview
- **Created**: 2025-05-09 04:16
- **Type**: Solar flare prediction model
- **Target**: C-class flares
- **Time Window**: 24 hours

## Description
Iteration on v2.900000000000001 model for C-class flares with 24h window

## Performance Metrics
- **final_training_accuracy**: 0.9804
- **final_training_loss**: 0.3861
- **epochs_trained**: 100
- **final_training_tss**: 0.9099


## Training Details
- **Architecture**: SolarKnowledge Transformer Model
- **Parameters**: 813,186
- **Precision**: torch.float32

## Hyperparameters
- **learning_rate**: 0.0001
- **weight_decay**: 0.0
- **batch_size**: 512
- **early_stopping_patience**: 5
- **early_stopping_metric**: loss
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
- **framework**: pytorch
- **input_shape**: (10, 9)
- **gradient_clipping**: True
- **max_grad_norm**: 1.0
- **random_seed**: 42
- **previous_version**: 2.900000000000001

## Version Control
- **Git Commit**: 9854d7a0a7b26ccc5e322918d559bd771cbe467a
- **Git Branch**: pytorch-rewrite

## Usage
```python
from SolarKnowledge_model import SolarKnowledge

# Load the model
model = SolarKnowledge()
model.load_model(
    input_shape=(10, 9),
    flare_class="C",
    w_dir="SolarKnowledge-v3.000000000000001-C-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
