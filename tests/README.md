# Testing Setup

This directory contains simplified tests for the core components of the EVEREST model training, ablation studies, and hyperparameter optimization.

## Test Structure

### Core Tests
- `test_training.py` - Tests for training configuration and experiment setup
- `test_ablation.py` - Tests for ablation study configuration and experiments
- `test_hpo.py` - Tests for hyperparameter optimization setup
- `test_imports.py` - Basic import tests for dependencies
- `test_forward_pass.py` - Model forward pass and training tests
- `test_metrics.py` - Custom metrics validation

### Legacy Tests
The directory also contains various analysis and calibration scripts from previous research iterations. These are kept for reference but are not part of the core CI pipeline.

## Running Tests

### Local Testing
```bash
# Run all core tests
pytest tests/test_*.py -v

# Run specific test file
pytest tests/test_training.py -v

# Run with coverage
pytest tests/test_*.py --cov=models --cov-report=term-missing
```

### CI Testing
The GitHub Actions workflow automatically runs:
1. Code formatting checks (black)
2. Linting (flake8)
3. Core functionality tests
4. Configuration validation for training/ablation/HPO

## Test Dependencies

Minimal dependencies are specified in `requirements-ci.txt`:
- pytest
- pytest-cov
- black
- flake8
- isort

Core ML dependencies (numpy, pandas, scikit-learn, tensorflow, torch, optuna) are installed separately in CI.

## Configuration Testing

The tests validate that:
- Training configurations are consistent and complete
- Ablation study setups are properly structured
- HPO search spaces and objectives are valid
- All required output directories can be created
- Experiment generation works correctly

This ensures that the research pipeline configurations are always in a valid state. 