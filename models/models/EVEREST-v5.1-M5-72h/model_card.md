# EVEREST Model v5.1

## Overview
- **Created**: 2025-05-29 01:26
- **Type**: Solar flare prediction model
- **Target**: M5-class flares
- **Time Window**: 72 hours

## Description
EVEREST Ablation Study - EVEREST model without evidential uncertainty (NIG head removed) (seed 0)

## Performance Metrics
- **accuracy**: 0.9966
- **TSS**: 0.6077


## Training Details
- **Architecture**: EVEREST Transformer Model
- **Parameters**: 813,574
- **Precision**: torch.float32

## Hyperparameters
- **input_shape**: (10, 9)
- **embed_dim**: 64
- **num_blocks**: 8
- **dropout**: 0.23876978467047777
- **ablation_variant**: no_evidential
- **ablation_seed**: 0
- **use_attention_bottleneck**: True
- **use_evidential**: False
- **use_evt**: True
- **use_precursor**: True
- **loss_weights**: {'focal': 0.9, 'evid': 0.0, 'evt': 0.1, 'prec': 0.05}

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
    w_dir="EVEREST-v5.1-M5-72h"
)

# Make predictions
predictions = model.predict(X_test)
```
