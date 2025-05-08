# SolarKnowledge PyTorch Implementation

This directory contains a complete PyTorch implementation of the SolarKnowledge model for solar flare prediction. The PyTorch implementation is designed to be faster and more efficient on modern hardware while maintaining the same accuracy as the original TensorFlow implementation.

## Files

- `SolarKnowledge_model_pytorch.py` - The PyTorch model implementation
- `SolarKnowledge_run_all_trainings_pytorch.py` - Script to train models
- `SolarKnowledge_run_all_tests_pytorch.py` - Script to test trained models
- `SolarKnowledge_Training.ipynb` - Jupyter notebook interface for training and testing

## Features

The PyTorch implementation includes several improvements:

1. **TensorFlow-compatible weight initialization** for consistent convergence
2. **Gradient clipping** to stabilize training
3. **Mixed precision training** for faster computation on GPUs
4. **Monte Carlo dropout** for uncertainty estimation
5. **Focal loss** for handling class imbalance
6. **Learning rate scheduling** for better convergence

## Requirements

To run the PyTorch implementation, you need:

```
torch>=2.0.0
numpy
matplotlib
pandas
scikit-learn
tqdm
seaborn
```

## Usage

### Command Line Interface

To train models for all flare classes and time windows:

```bash
python SolarKnowledge_run_all_trainings_pytorch.py
```

To train a specific model:

```bash
python SolarKnowledge_run_all_trainings_pytorch.py --specific-flare C --specific-window 24
```

Available options:
- `--version` or `-v`: Model version identifier (auto-incremented by default)
- `--description` or `-d`: Description of the model
- `--specific-flare` or `-f`: Train only for a specific flare class (C, M, or M5)
- `--specific-window` or `-w`: Train only for a specific time window (24, 48, or 72)
- `--compare`: Compare all models after training
- `--no-auto-increment`: Do not auto-increment version

### Testing Models

To test trained models:

```bash
python SolarKnowledge_run_all_tests_pytorch.py
```

Available options:
- `--timestamp` or `-t`: Specific model timestamp to test
- `--latest`: Test the latest model version
- `--version` or `-v`: Test a specific model version
- `--mc-passes`: Number of Monte Carlo dropout passes (default: 20)
- `--no-plots`: Skip generating uncertainty plots

### Jupyter Notebook

For an interactive experience, you can use the `SolarKnowledge_Training.ipynb` notebook, which provides a user-friendly interface for:

1. Training models with customizable parameters
2. Testing models and generating visualizations
3. Comparing model performance across different configurations

## Performance Comparison with TensorFlow

The PyTorch implementation is significantly faster than the TensorFlow implementation:

- Training speed: ~6s/epoch (PyTorch) vs ~21-23s/epoch (TensorFlow)
- Compatible weight initialization ensures similar convergence

## Troubleshooting

If you encounter issues with convergence, try:

1. Ensuring weight initialization is properly configured
2. Adjusting the learning rate or batch size
3. Checking that class weights are appropriate for your dataset

## Citing this Work

If you use this implementation in your research, please cite:

```
@software{SolarKnowledge_PyTorch,
  author = {Zilinskas, Antanas},
  title = {SolarKnowledge: A Transformer-based Model for Solar Flare Prediction},
  year = {2023},
  url = {https://github.com/yourusername/SolarKnowledge},
}
``` 