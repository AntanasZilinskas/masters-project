# SolarKnowledge Model v3.300000000000001

## Overview
- **Created**: 2025-05-09 08:20
- **Type**: Solar flare prediction model
- **Target**: C-class flares
- **Time Window**: 24 hours

## Description
Iteration on v3.200000000000001 model for C-class flares with 24h window

## Performance Metrics
- **final_training_accuracy**: 0.7945
- **final_training_loss**: 0.1247
- **epochs_trained**: 200
- **final_training_tss**: 0.5270


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
- **lr_scheduler**: {'type': 'ReduceLROnPlateau', 'monitor': 'loss', 'factor': 0.5, 'patience': 3, 'min_lr': 1e-06}
- **epochs**: 200
- **num_transformer_blocks**: 6
- **embed_dim**: 128
- **num_heads**: 4
- **ff_dim**: 256
- **dropout_rate**: 0.2
- **focal_loss**: True
- **focal_loss_alpha**: 0.25
- **focal_loss_gamma**: 2.0
- **class_weights**: {0: 1.0, 1: 0.5554805656124052}
- **framework**: pytorch
- **input_shape**: (10, 9)
- **gradient_clipping**: True
- **max_grad_norm**: 1.0
- **random_seed**: 42
- **regularization**: {'l1': 1e-05, 'l2': 0.0001}
- **previous_version**: 3.200000000000001

## Version Control
- **Git Commit**: e86d19d349bc94c6b192ac39f05695dcb597378b
- **Git Branch**: pytorch-rewrite

## Usage
```python
from SolarKnowledge_model import SolarKnowledge

# Load the model
model = SolarKnowledge()
model.load_model(
    input_shape=(10, 9),
    flare_class="C",
    w_dir="SolarKnowledge-v3.300000000000001-C-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
