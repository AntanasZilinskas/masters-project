# SolarKnowledge Model v1.2

## Overview
- **Created**: 2025-04-18 17:09
- **Type**: Solar flare prediction model
- **Target**: C-class flares
- **Time Window**: 72 hours

## Description
Retraining the best performing model using mixed precision

## Performance Metrics
- **final_training_accuracy**: 0.9884
- **final_training_loss**: 0.0451
- **epochs_trained**: 169


## Training Details
- **Architecture**: SolarKnowledge Transformer Model
- **Parameters**: 2,000,002
- **Precision**: <DTypePolicy "float32">

## Hyperparameters
- **learning_rate**: 0.0001
- **batch_size**: 512
- **early_stopping_patience**: 5
- **epochs**: 200
- **num_transformer_blocks**: 6
- **embed_dim**: 128
- **num_heads**: 4
- **ff_dim**: 256
- **dropout_rate**: 0.2
- **previous_version**: 1.1

## Version Control
- **Git Commit**: 365e7dd00cfc559533d8e855b59b95138532eb9a
- **Git Branch**: main

## Usage
```python
from SolarKnowledge_model import SolarKnowledge

# Load the model
model = SolarKnowledge()
model.load_model(
    input_shape=(10, 9), 
    flare_class="C", 
    w_dir="SolarKnowledge-v1.2-C-72h"
)

# Make predictions
predictions = model.predict(X_test)
```
