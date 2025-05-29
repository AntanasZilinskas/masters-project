# EVEREST Model v5.0

## Overview
- **Created**: 2025-05-29 01:14
- **Type**: Solar flare prediction model
- **Target**: M5-class flares
- **Time Window**: 72 hours

## Description
EVEREST Ablation Study - Full EVEREST model with all components (baseline) (seed 3)

## Performance Metrics
- **accuracy**: 0.9965
- **TSS**: 0.5873


## Training Details
- **Architecture**: EVEREST Transformer Model
- **Parameters**: 814,090
- **Precision**: torch.float32

## Hyperparameters
- **input_shape**: (10, 9)
- **embed_dim**: 64
- **num_blocks**: 8
- **dropout**: 0.23876978467047777
- **ablation_variant**: full_model
- **ablation_seed**: 3
- **use_attention_bottleneck**: True
- **use_evidential**: True
- **use_evt**: True
- **use_precursor**: True
- **loss_weights**: {'focal': 0.8, 'evid': 0.1, 'evt': 0.1, 'prec': 0.05}

## Version Control
- **Git Commit**: c7ea6cf76813c99df1dd974ebc9da3aaa8d5a4b2
- **Git Branch**: pytorch-rewrite

## Usage
```python
from EVEREST_model import EVEREST

# Load the model
model = EVEREST()
model.load_model(
    input_shape=(10, 9),
    flare_class="M5",
    w_dir="EVEREST-v5.0-M5-72h"
)

# Make predictions
predictions = model.predict(X_test)
```
