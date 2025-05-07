# SolarKnowledge Model v1.3

## Overview
- **Created**: 2025-04-19 20:57
- **Type**: Solar flare prediction model
- **Target**: M5-class flares
- **Time Window**: 72 hours

## Description
Retraining the operational model using an older version of Tensorflow for mixed precision support

## Performance Metrics
- **final_training_accuracy**: 0.9995
- **final_training_loss**: 0.0028
- **epochs_trained**: 94


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
- **previous_version**: 1.2

## Version Control
- **Git Commit**: ef15a595fc506cca97bd285f0a648ad137252ed0
- **Git Branch**: main

## Usage
```python
from SolarKnowledge_model import SolarKnowledge

# Load the model
model = SolarKnowledge()
model.load_model(
    input_shape=(10, 9), 
    flare_class="M5", 
    w_dir="SolarKnowledge-v1.3-M5-72h"
)

# Make predictions
predictions = model.predict(X_test)
```
