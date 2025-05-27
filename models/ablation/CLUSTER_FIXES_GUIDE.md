# EVEREST Ablation Study - Cluster Fixes Guide

## Problem Diagnosis

The ablation study was failing on the Imperial RCS cluster due to **import path issues** that were different from how the HPO study was structured. The key problems were:

### 1. Module Import vs Direct Script Execution
- **HPO approach (working)**: Called scripts directly like `python models/hpo/run_hpo.py`
- **Ablation approach (broken)**: Used module imports like `python -m ablation.trainer`

### 2. Relative Import Issues
The ablation trainer used relative imports (`from .config import`) which only work when the module is properly installed as a package, but fail when called directly as a script.

### 3. Python Path Configuration
The cluster environment didn't have the correct Python path setup for the module-based imports.

## Fixes Applied

### 1. Fixed Cluster Script (`submit_array_fixed.pbs`)

**Changed from module imports to direct script execution:**
```bash
# OLD (broken):
python -m ablation.trainer --variant $variant_or_seq --seed $seed

# NEW (fixed):
python models/ablation/trainer.py --variant $variant_or_seq --seed $seed
```

**Added import testing:**
```bash
# Test imports before running experiments
echo "Testing imports..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import sys; sys.path.append('models'); from solarknowledge_ret_plus import RETPlusWrapper; print('✅ EVEREST imports successful')"
python -c "import sys; sys.path.append('models'); from utils import get_training_data; print('✅ Utils imports successful')"
```

### 2. Fixed Import Structure (`trainer.py`)

**Added fallback import handling:**
```python
# Import config - handle both direct script execution and module import
try:
    from .config import (
        OPTIMAL_HYPERPARAMS, FIXED_ARCHITECTURE, PRIMARY_TARGET,
        TRAINING_CONFIG, ABLATION_VARIANTS, SEQUENCE_LENGTH_VARIANTS,
        OUTPUT_CONFIG, get_variant_config, get_sequence_config,
        get_experiment_name
    )
except ImportError:
    # Direct script execution - import from same directory
    from config import (
        OPTIMAL_HYPERPARAMS, FIXED_ARCHITECTURE, PRIMARY_TARGET,
        TRAINING_CONFIG, ABLATION_VARIANTS, SEQUENCE_LENGTH_VARIANTS,
        OUTPUT_CONFIG, get_variant_config, get_sequence_config,
        get_experiment_name
    )
```

### 3. Created Test Script (`test_trainer.py`)

Added comprehensive testing to verify all imports and functionality work correctly before cluster submission.

## Verification

The fixes have been tested locally and all tests pass:

```bash
cd models/ablation
python test_trainer.py
```

Output:
```
🔬 EVEREST Ablation Trainer Test Suite
==================================================
🧪 Testing imports...
✅ EVEREST model imports successful
✅ Utils imports successful
✅ Config imports successful
✅ Trainer imports successful

🧪 Testing trainer creation...
✅ Trainer creation successful

🧪 Testing data loading...
✅ Training data loaded: 709447 samples
✅ Testing data loaded: 71729 samples

📊 Test Results: 3/3 tests passed
🎉 All tests passed! Trainer is ready for cluster execution.
```

## Usage Instructions

### 1. Submit Fixed Array Job

```bash
# Cancel any existing jobs first
qdel <job_id>

# Submit the fixed array job
qsub models/ablation/cluster/submit_array_fixed.pbs
```

### 2. Monitor Progress

```bash
# Check job status
qstat -u $USER

# Check logs
ls logs/ablation_array_*

# View specific log
tail -f logs/ablation_array_<jobid>_<arrayindex>.log
```

### 3. Local Testing (Optional)

Test individual experiments locally:
```bash
cd models/ablation

# Test single component ablation
python trainer.py --variant no_evidential --seed 0

# Test sequence length ablation  
python trainer.py --variant full_model --seed 0 --sequence seq_15
```

## Key Differences from HPO

| Aspect | HPO (Working) | Ablation (Fixed) |
|--------|---------------|------------------|
| Script Call | `python models/hpo/run_hpo.py` | `python models/ablation/trainer.py` |
| Import Style | Direct imports | Fallback imports (relative → direct) |
| Module Structure | Single script | Package with fallback |
| Path Setup | Simple `sys.path.append` | Robust path handling |

## Expected Results

The fixed array job will:
- Run 60 total experiments (35 component + 25 sequence ablations)
- Distribute across 10 array jobs (6-7 experiments each)
- Create proper log files for each array job
- Save results to `models/ablation/results/`
- Complete successfully without import errors

## Troubleshooting

If issues persist:

1. **Check import test output** in the log files
2. **Verify Python environment** is activated correctly
3. **Check data file paths** exist on the cluster
4. **Monitor GPU availability** during execution

The key insight is that **cluster environments require robust import handling** and **direct script execution is more reliable than module imports** for complex packages. 