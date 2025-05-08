# SolarKnowledge Model v1.6

## Overview
- **Created**: 2025-05-08 04:57
- **Type**: Solar flare prediction model
- **Target**: C-class flares
- **Time Window**: 24 hours

## Description
PyTorch model with TensorFlow-compatible initialization and focal loss

## Performance Metrics
- **final_training_accuracy**: 0.8564
- **final_training_loss**: 0.1735
- **epochs_trained**: 100
- **final_training_tss**: 0.6622


## Training Details
- **Architecture**: SolarKnowledge Transformer Model
- **Parameters**: 813,186
- **Precision**: torch.float32

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
- **framework**: pytorch
- **weight_initialization**: tf_compatible
- **gradient_clipping**: True
- **max_grad_norm**: 1.0
- **input_shape**: (10, 9)
- **previous_version**: 1.5

## Version Control
- **Git Commit**: 32c7c105f16dc2fd8b8baadb45060d62cbd8acc8
- **Git Branch**: pytorch-rewrite

## Usage
```python
from SolarKnowledge_model import SolarKnowledge

# Load the model
model = SolarKnowledge()
model.load_model(
    input_shape=(10, 9),
    flare_class="C",
    w_dir="SolarKnowledge-v1.6-C-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
