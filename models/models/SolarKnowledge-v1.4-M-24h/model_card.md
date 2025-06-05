# SolarKnowledge Model v1.4

## Overview
- **Created**: 2025-05-08 04:47
- **Type**: Solar flare prediction model
- **Target**: M-class flares
- **Time Window**: 24 hours

## Description
Iteration on v1.3 model for M-class flares with 24h window

## Performance Metrics
- **final_training_accuracy**: 0.9664
- **final_training_loss**: 0.0445
- **epochs_trained**: 75
- **final_training_tss**: 0.9331


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
- **class_weights**: {0: 1.0, 1: 7.184649122807018}
- **previous_version**: 1.3

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
    flare_class="M",
    w_dir="SolarKnowledge-v1.4-M-24h"
)

# Make predictions
predictions = model.predict(X_test)
```
