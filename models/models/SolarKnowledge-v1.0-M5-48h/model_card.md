# SolarKnowledge Model v1.0

## Overview
- **Created**: 2025-04-18 08:05
- **Type**: Solar flare prediction model
- **Target**: M5-class flares
- **Time Window**: 48 hours

## Description
Retraining the best performing model

## Performance Metrics
- **final_training_accuracy**: 0.9989
- **final_training_loss**: 0.0120
- **epochs_trained**: 68


## Training Details
- **Architecture**: SolarKnowledge Transformer Model
- **Parameters**: 1,999,746
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
    flare_class="M5", 
    w_dir="SolarKnowledge-v1.0-M5-48h"
)

# Make predictions
predictions = model.predict(X_test)
```
