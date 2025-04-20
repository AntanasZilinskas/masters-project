# SHARP-2023-CI-Sample Dataset

This directory contains a small subset of the full SHARP dataset used in Abduallah et al. (2023), specifically designed for CI testing. It allows for quick smoke tests of the model training pipeline without requiring the full dataset.

## Source
Data is derived from:
> Abduallah, Y., Wang, J.T.L., Wang, H. et al. Operational prediction of solar flares using a transformer-based framework. Sci Rep 13, 13665 (2023). https://doi.org/10.1038/s41598-023-40884-1

## Contents

- 30 tensor sequences (15 per class)
- Each sequence is 100 timesteps × 14 features
- Total size: < 5MB

## Purpose

This dataset is intended for:
1. CI smoke testing
2. Quick model validation during development
3. Testing data loading and preprocessing pipelines

## Generation

This dataset was created by randomly sampling from the full SHARP dataset used in the paper and ensuring equal class distribution. The script used to generate this dataset can be found in `scripts/generate_tiny_sample.py`.

## Usage

To use this dataset in your workflow:

```python
from solar_knowledge.data import load_tiny_dataset

# Load the dataset
X_train, y_train = load_tiny_dataset()

# Use for model training/testing
model.fit(X_train, y_train, epochs=1)
```
