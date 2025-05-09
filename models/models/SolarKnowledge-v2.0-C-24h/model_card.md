# SolarKnowledge Model v2.0

## Overview
- **Created**: 2025-05-08 21:38
- **Type**: Solar flare prediction model
- **Target**: C-class flares
- **Time Window**: 24 hours

## Description
Using One Cycle LR and SWA for faster convergence

## Performance Metrics
- **final_training_accuracy**: 0.9705
- **final_training_loss**: 0.0206
- **epochs_trained**: 114
- **final_training_tss**: 0.9411


## Training Details
- **Architecture**: SolarKnowledge Transformer Model
- **Parameters**: 813,186
- **Precision**: torch.float32

## Hyperparameters
- **learning_rate**: 0.0003
- **weight_decay**: 0.0
- **batch_size**: 512
- **early_stopping_patience**: 5
- **epochs**: 300
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
- **weight_initialization**: tf_compatible
- **gradient_clipping**: True
- **max_grad_norm**: 1.0
- **input_shape**: (10, 9)
- **scheduler**: reduce_on_plateau
- **scheduler_params**: {'mode': 'min', 'factor': 0.5, 'patience': 5, 'min_lr': 1e-06, 'T_0': 5, 'T_mult': 2, 'eta_min': 1e-07}
- **use_batch_norm**: True
- **optimizer**: Adam
- **random_seed**: 42
- **previous_version**: 1.9

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
    w_dir="SolarKnowledge-v2.0-C-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
