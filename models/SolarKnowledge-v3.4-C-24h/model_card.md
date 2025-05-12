# SolarKnowledge Model v3.4

## Overview
- **Created**: 2025-05-09 07:45
- **Type**: Solar flare prediction model
- **Target**: C-class flares
- **Time Window**: 24 hours

## Description
A6000-optimised config: deeper, wider, faster

## Performance Metrics
- **final_training_accuracy**: 0.9701
- **final_training_loss**: 0.1355
- **epochs_trained**: 200
- **final_training_tss**: 0.8069


## Training Details
- **Architecture**: SolarKnowledge Transformer Model
- **Parameters**: 6,354,306
- **Precision**: torch.float32

## Hyperparameters
- **learning_rate**: 0.0003
- **weight_decay**: 0.0001
- **batch_size**: 2048
- **early_stopping_patience**: 15
- **early_stopping_metric**: loss
- **epochs**: 200
- **num_transformer_blocks**: 8
- **embed_dim**: 256
- **num_heads**: 8
- **ff_dim**: 1024
- **dropout_rate**: 0.3
- **focal_loss**: True
- **focal_loss_alpha**: 0.25
- **focal_loss_gamma**: 2.0
- **framework**: pytorch
- **gradient_clipping**: True
- **max_grad_norm**: 1.0
- **input_shape**: (10, 9)
- **lr_scheduler**: {'type': 'cosine_with_restarts', 'T_0': 10, 'T_mult': 2, 'eta_min': 1e-06}
- **regularization**: {'l1': 1e-05, 'l2': 0.0001}
- **optimizer**: AdamW
- **random_seed**: 42
- **previous_version**: 3.3

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
    w_dir="SolarKnowledge-v3.4-C-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
