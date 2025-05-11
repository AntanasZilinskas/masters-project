# SolarKnowledge Model v2.3000000000000003

## Overview
- **Created**: 2025-05-09 02:44
- **Type**: Solar flare prediction model
- **Target**: C-class flares
- **Time Window**: 24 hours

## Description
Test training run

## Performance Metrics
- **final_training_accuracy**: 0.7256
- **final_training_loss**: 0.1497
- **epochs_trained**: 1
- **final_training_tss**: 0.2385


## Training Details
- **Architecture**: SolarKnowledge Transformer Model
- **Parameters**: 813,186
- **Precision**: torch.float32

## Hyperparameters
- **learning_rate**: 0.0003
- **weight_decay**: 0.0
- **batch_size**: 512
- **early_stopping_patience**: 5
- **epochs**: 1
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
- **weight_initialization**: tf_compatible
- **gradient_clipping**: True
- **max_grad_norm**: 1.0
- **input_shape**: (10, 9)
- **scheduler**: reduce_on_plateau
- **scheduler_params**: {'mode': 'min', 'factor': 0.5, 'patience': 5, 'min_lr': 1e-06, 'T_0': 5, 'T_mult': 2, 'eta_min': 1e-07}
- **use_batch_norm**: True
- **optimizer**: Adam
- **random_seed**: 42
- **previous_version**: 2.2

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
    w_dir="SolarKnowledge-v2.3000000000000003-C-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
