# SolarKnowledge Model v1.0

## Overview
- **Created**: 2025-05-08 19:11
- **Type**: Solar flare prediction model
- **Target**: C-class flares
- **Time Window**: 24 hours

## Description
Initial model for C-class flares with 24h prediction window

## Performance Metrics
- **final_training_accuracy**: 0.9229
- **final_training_loss**: 0.1519
- **epochs_trained**: 300
- **final_training_tss**: 0.8201


## Training Details
- **Architecture**: SolarKnowledge Transformer Model
- **Parameters**: 4,274,378
- **Precision**: torch.float32

## Hyperparameters
- **learning_rate**: 5e-05
- **weight_decay**: 0.0001
- **batch_size**: 128
- **early_stopping_patience**: 10
- **epochs**: 300
- **num_transformer_blocks**: 8
- **embed_dim**: 256
- **num_heads**: 8
- **ff_dim**: 512
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
- **scheduler**: cosine_with_restarts
- **scheduler_params**: {'T_0': 10, 'T_mult': 2, 'min_lr': 1e-07, 'eta_min': 1e-07}
- **use_batch_norm**: True
- **optimizer**: AdamW
- **random_seed**: 42

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
