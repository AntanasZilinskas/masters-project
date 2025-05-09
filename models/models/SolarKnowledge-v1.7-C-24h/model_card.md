# SolarKnowledge Model v1.7000000000000002

## Overview
- **Created**: 2025-05-08 06:34
- **Type**: Solar flare prediction model
- **Target**: C-class flares
- **Time Window**: 24 hours

## Description
Enhanced PyTorch model with batch norm, AdamW and cosine annealing

## Performance Metrics
- **final_training_accuracy**: 0.9800
- **final_training_loss**: 0.5284
- **epochs_trained**: 300
- **final_training_tss**: 0.9537


## Training Details
- **Architecture**: SolarKnowledge Transformer Model
- **Parameters**: 4,332,426
- **Precision**: torch.float32

## Hyperparameters
- **learning_rate**: 5e-05
- **weight_decay**: 0.0001
- **batch_size**: 512
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
- **scheduler_params**: {'T_0': 10, 'T_mult': 2, 'min_lr': 1e-07}
- **use_batch_norm**: True
- **optimizer**: AdamW
- **random_seed**: 42
- **previous_version**: 1.6

## Version Control
- **Git Commit**: ef830282d47061fe734aa9d99f65f98ba049e29d
- **Git Branch**: pytorch-rewrite

## Usage
```python
from SolarKnowledge_model import SolarKnowledge

# Load the model
model = SolarKnowledge()
model.load_model(
    input_shape=(10, 9),
    flare_class="C",
    w_dir="SolarKnowledge-v1.7000000000000002-C-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
