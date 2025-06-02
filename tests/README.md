# Testing Setup

This directory contains simplified tests for the core components of the EVEREST model training, ablation studies, and hyperparameter optimization. The tests are designed to work across multiple branches with different ML frameworks.

## Multi-Branch Support

The CI system supports multiple branches:
- **`main`** - TensorFlow-based implementation
- **`pytorch-rewrite`** - PyTorch-based implementation  
- **`develop`** - Development branch (TensorFlow)

Tests automatically adapt to the available framework and skip unavailable components gracefully.

## Test Structure

### Core Tests
- `test_training.py` - Tests for training configuration and experiment setup
- `test_ablation.py` - Tests for ablation study configuration and experiments
- `test_hpo.py` - Tests for hyperparameter optimization setup
- `test_imports.py` - Basic import tests for dependencies
- `test_forward_pass.py` - Model forward pass and training tests
- `test_metrics.py` - Custom metrics validation

### Framework Compatibility Tests
Each test file includes framework compatibility tests that:
- Detect the current branch automatically
- Test appropriate ML framework (TensorFlow vs PyTorch)
- Skip unavailable components gracefully
- Validate framework-specific functionality

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

# Test specific branch compatibility
GITHUB_REF=refs/heads/pytorch-rewrite pytest tests/test_*.py -v
```

### CI Testing
The GitHub Actions workflow automatically runs on all supported branches:
1. **Code Quality Checks:**
   - Black formatting (88-character line length)
   - Flake8 linting (critical errors only)
   - Basic pre-commit hooks

2. **Core Functionality Tests:**
   - Import validation
   - Configuration testing
   - Framework compatibility

3. **Branch-Specific Tests:**
   - **Main/Develop:** TensorFlow functionality tests
   - **PyTorch-rewrite:** PyTorch functionality tests
   - **All branches:** Framework-agnostic core logic

4. **Configuration Validation:**
   - Training configurations (if available)
   - Ablation study setups (if available)
   - HPO search spaces (if available)

## Test Dependencies

Minimal dependencies are specified in `requirements-ci.txt`:
- pytest
- pytest-cov
- black
- flake8
- isort

Core ML dependencies are installed automatically:
- numpy, pandas, scikit-learn (all branches)
- tensorflow (main, develop branches)
- torch (pytorch-rewrite branch)
- optuna (HPO testing)

## Framework Compatibility

### TensorFlow Branches (main, develop)
Tests validate:
- TensorFlow model creation and operations
- Keras-based training workflows
- TensorFlow-specific configurations

### PyTorch Branch (pytorch-rewrite)
Tests validate:
- PyTorch tensor operations
- PyTorch model definitions
- PyTorch-specific training loops

### Framework-Agnostic Components
- Configuration validation
- Experiment generation
- Output directory management
- Mathematical operations (numpy-based)

## Configuration Testing

The tests validate that:
- Training configurations are consistent and complete
- Ablation study setups are properly structured
- HPO search spaces and objectives are valid
- All required output directories can be created
- Experiment generation works correctly
- Framework compatibility is maintained

### Graceful Degradation
Tests use graceful degradation when components are unavailable:
- Missing config modules → Skip with informative message
- Missing frameworks → Skip framework-specific tests
- Import errors → Skip with pytest.skip()

This ensures that tests pass even when running on branches with different structures or dependencies.

## Branch-Specific Features

### Main Branch
- Full TensorFlow integration
- Complete training/ablation/HPO configs
- Legacy analysis scripts

### PyTorch-Rewrite Branch
- PyTorch model implementations
- Framework-specific optimizations
- Potentially different config structures

### Cross-Branch Consistency
- Shared test logic for core functionality
- Common configuration validation patterns
- Framework-agnostic mathematical operations

This design ensures that the research pipeline configurations remain valid and testable across all development branches while allowing for framework-specific optimizations. 