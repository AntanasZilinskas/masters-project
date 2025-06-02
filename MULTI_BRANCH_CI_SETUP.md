# Multi-Branch CI Setup Guide

## Overview

The EVEREST repository now supports a robust multi-branch CI system that works seamlessly across different ML frameworks while maintaining consistent quality standards and research pipeline validation.

## Supported Branches

| Branch | Framework | Description | CI Status |
|--------|-----------|-------------|-----------|
| `main` | TensorFlow | Production-ready TensorFlow implementation | ✅ Full CI |
| `pytorch-rewrite` | PyTorch | PyTorch-based implementation | ✅ Full CI |
| `develop` | TensorFlow | Development/experimental features | ✅ Full CI |

## How It Works

### 🔍 **Automatic Branch Detection**
The CI system automatically detects which branch is being tested and adapts accordingly:

```yaml
# CI automatically detects branch and runs appropriate tests
BRANCH_NAME=${GITHUB_REF#refs/heads/}
if [[ "$BRANCH_NAME" == "pytorch-rewrite" ]]; then
  # Run PyTorch-specific tests
elif [[ "$BRANCH_NAME" == "main" ]] || [[ "$BRANCH_NAME" == "develop" ]]; then
  # Run TensorFlow-specific tests
fi
```

### 🧪 **Framework-Adaptive Testing**
Tests automatically adapt to the available framework:

```python
# Example from test_training.py
def test_branch_compatibility():
    branch_name = os.environ.get('GITHUB_REF', '').replace('refs/heads/', '')
    
    if 'pytorch' in branch_name:
        import torch
        assert torch.__version__ is not None
        print(f"✓ PyTorch {torch.__version__} detected")
    else:
        import tensorflow as tf
        assert tf.__version__ is not None
        print(f"✓ TensorFlow {tf.__version__} detected")
```

### 🛡️ **Graceful Degradation**
Missing components are handled gracefully:

```python
# Example from test_ablation.py
def test_ablation_config_import():
    try:
        from ablation.config import validate_ablation_config
        validate_ablation_config()
        print('✓ Ablation config valid.')
    except ImportError:
        pytest.skip("Ablation config not available in this branch")
```

## CI Workflow Features

### 🚀 **Multi-Branch Triggers**
```yaml
on:
  push:
    branches: [ main, develop, pytorch-rewrite ]
  pull_request:
    branches: [ main, pytorch-rewrite ]
```

### 🧬 **Framework-Specific Steps**
The CI includes dedicated steps for each framework:

#### TensorFlow Branches (main, develop)
- TensorFlow operations validation
- Keras workflow testing
- TensorFlow-specific model tests

#### PyTorch Branch (pytorch-rewrite)
- PyTorch tensor operations
- PyTorch model definition tests
- PyTorch-specific training validation

#### All Branches
- Code quality checks (black, flake8)
- Configuration validation
- Framework-agnostic core logic
- HPO compatibility testing

## Development Workflow

### 🔄 **For Main Branch (TensorFlow)**
```bash
# Develop on main branch
git checkout main

# Make changes
# ...

# Tests automatically validate TensorFlow functionality
git push origin main
```

### 🔥 **For PyTorch Rewrite**
```bash
# Develop on pytorch-rewrite branch
git checkout pytorch-rewrite

# Make PyTorch-specific changes
# ...

# Tests automatically validate PyTorch functionality
git push origin pytorch-rewrite
```

### 🧪 **For Development Branch**
```bash
# Experimental features
git checkout develop

# Add experimental features
# ...

# Full TensorFlow CI validation
git push origin develop
```

## Local Testing

### 🏠 **Test Current Branch**
```bash
# Run tests adapted to current branch
pytest tests/test_*.py -v
```

### 🎯 **Test Specific Branch Compatibility**
```bash
# Simulate main branch
GITHUB_REF=refs/heads/main pytest tests/test_*.py -v

# Simulate pytorch-rewrite branch  
GITHUB_REF=refs/heads/pytorch-rewrite pytest tests/test_*.py -v

# Simulate develop branch
GITHUB_REF=refs/heads/develop pytest tests/test_*.py -v
```

### 🔍 **Test Individual Components**
```bash
# Test training configs (adapts to branch)
pytest tests/test_training.py -v

# Test ablation studies (skips if unavailable)
pytest tests/test_ablation.py -v

# Test HPO (works with both frameworks)
pytest tests/test_hpo.py -v
```

## Configuration Management

### 📋 **Shared Configurations**
Core configurations that work across all branches:
- Experiment definitions
- Hyperparameter spaces
- Evaluation metrics
- Output directory structures

### 🔧 **Branch-Specific Configurations**
Configurations that may differ between branches:
- Model architectures (TensorFlow vs PyTorch)
- Training loops (framework-specific)
- Optimization strategies

### ✅ **Validation Strategy**
```python
# Configurations are validated when available
try:
    from training.config import validate_config
    validate_config()
    print("✓ Training config validated")
except ImportError:
    print("⚠️ Training config not available (skipped)")
```

## Benefits

### 🎯 **For Researchers**
- **Consistent validation** across framework implementations
- **Experiment reproducibility** regardless of framework choice
- **Configuration consistency** ensures comparable results

### 🛠️ **For Developers**
- **Parallel development** on multiple frameworks
- **Framework migration** support with continuous validation
- **Quality assurance** maintained across all branches

### 🏃 **For CI/CD**
- **Fast feedback** with framework-specific optimizations
- **Reliable testing** that adapts to branch differences
- **Reduced maintenance** with intelligent fallback behavior

## Troubleshooting

### ❌ **Common Issues**

#### Import Errors
```bash
# Issue: Module not found
ImportError: No module named 'training.config'

# Solution: This is expected behavior - test will skip gracefully
# The warning message will indicate the component is unavailable
```

#### Framework Conflicts
```bash
# Issue: Wrong framework installed
ImportError: No module named 'torch'

# Solution: Install appropriate framework for branch
pip install torch  # for pytorch-rewrite
pip install tensorflow  # for main/develop
```

#### Configuration Mismatches
```bash
# Issue: Configuration validation fails
AssertionError: Expected configuration not found

# Solution: Check branch-specific config structure
# Some configurations may differ between framework implementations
```

### ✅ **Best Practices**

1. **Always test locally** before pushing to ensure compatibility
2. **Use descriptive commit messages** indicating framework-specific changes
3. **Update documentation** when adding branch-specific features
4. **Validate configurations** work across relevant branches

## Migration Path

### 🔄 **Adding New Frameworks**
To add support for a new framework:

1. **Create new branch** following naming convention
2. **Update CI triggers** to include new branch
3. **Add framework detection** in test files
4. **Implement framework-specific tests**
5. **Update documentation**

### 🔧 **Modifying Existing Branches**
When making changes to existing branches:

1. **Test locally** with branch simulation
2. **Ensure backward compatibility** where possible
3. **Update tests** if configuration structure changes
4. **Validate CI passes** on all affected branches

This multi-branch CI system ensures that research can proceed efficiently across different ML frameworks while maintaining the highest standards of code quality and experimental reproducibility. 