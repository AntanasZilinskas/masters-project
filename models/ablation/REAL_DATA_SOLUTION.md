# EVEREST Ablation Study - Real Data Solution

## Overview

This solution uses **real SHARP solar flare data** via the existing `get_training_data()` and `get_testing_data()` functions, with pandas compatibility fixes for the Imperial RCS cluster.

## Key Features

✅ **Real Data**: Uses actual SHARP magnetogram data  
✅ **Exact Environment**: Uses same setup as working training scripts  
✅ **HPO Pattern**: Follows the exact same structure as working HPO study  
✅ **Comprehensive**: 60 experiments across component and sequence ablations  
✅ **Cluster Optimized**: 10 parallel array jobs with 6 experiments each  

## Files

### Core Scripts
- `run_ablation_hpo_style.py` - Main ablation runner (follows HPO pattern)
- `cluster/submit_real_data_fixed.pbs` - Production array job submission
- `cluster/test_real_data.pbs` - Test script for validation

### Cluster Submission
```bash
# Test first (recommended)
qsub cluster/test_real_data.pbs

# Then submit full study
qsub cluster/submit_real_data_fixed.pbs
```

## Experiment Design

### Component Ablations (35 experiments)
- **full_model**: Complete EVEREST with all components
- **no_evidential**: Remove evidential (NIG) head
- **no_evt**: Remove EVT (GPD) head  
- **mean_pool**: Replace attention pooling with mean pooling
- **cross_entropy**: Use standard BCE instead of focal loss
- **no_precursor**: Remove precursor prediction head
- **fp32_training**: Use FP32 instead of mixed precision

### Sequence Ablations (25 experiments)
- **seq_5**: 5-timestep sequences (vs default 10)
- **seq_7**: 7-timestep sequences
- **seq_10**: 10-timestep sequences (baseline)
- **seq_15**: 15-timestep sequences
- **seq_20**: 20-timestep sequences

Each variant runs with 5 different random seeds (0-4) for statistical robustness.

## Environment Setup

Uses the exact same environment setup as your working training scripts:

1. **Load anaconda module**: `module load anaconda3/personal`
2. **Activate everest_env**: `source activate everest_env`
3. **Set environment variables**: Standard CUDA and PYTHONPATH setup
4. **Pandas works**: No compatibility issues in everest_env

```bash
# Environment setup (same as working training scripts)
module load anaconda3/personal
source activate everest_env
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH=$PBS_O_WORKDIR:$PYTHONPATH
```

## Data Loading

Uses the same proven data loading pattern as the working HPO study:

```python
# Same imports and path setup as HPO
from models.utils import get_training_data, get_testing_data

# Load real SHARP data
X_train, y_train = get_training_data('72', 'M5')  # 72h M5-class
X_test, y_test = get_testing_data('72', 'M5')
```

## Job Distribution

**Array Job Structure**: 10 parallel jobs, each running 6 experiments

| Job | Experiments | Variants |
|-----|-------------|----------|
| 1   | 1-6         | full_model (seeds 0-4) + no_evidential (seed 0) |
| 2   | 7-12        | no_evidential (seeds 1-4) + no_evt (seed 0-1) |
| ... | ...         | ... |
| 10  | 55-60       | seq_20 (seeds 0-4) + buffer |

## Resource Requirements

- **Walltime**: 12 hours per job
- **GPU**: RTX6000 (required for mixed precision)
- **Memory**: 32GB (sufficient for SHARP data)
- **CPUs**: 8 cores (data loading parallelism)

## Expected Outputs

Each experiment saves:
- **Model weights**: `models/ablation/trained_models/`
- **Results**: `models/ablation/results/`
- **Logs**: `models/ablation/logs/`

Results include:
- Final metrics (TSS, F1, accuracy, precision, recall)
- Training history
- Model metadata
- Timing information

## Validation Steps

1. **Test pandas**: `python -c "import pandas"`
2. **Test data loading**: Load M5/72h training and test data
3. **Test GPU**: Verify CUDA availability
4. **Test ablation**: Run 1-epoch training

## Troubleshooting

### Pandas Import Error
```
ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
```
**Solution**: The script automatically tries system libstdc++ fallback

### Data Loading Error
```
FileNotFoundError: No such file or directory: 'data/...'
```
**Solution**: Ensure you're in the project root directory (`masters-project/`)

### GPU Not Available
```
CUDA not available
```
**Solution**: Check PBS resource allocation includes `ngpus=1:gpu_type=RTX6000`

## Comparison with Synthetic Approach

| Aspect | Real Data | Synthetic Data |
|--------|-----------|----------------|
| **Scientific Validity** | ✅ Actual solar data | ⚠️ Simulated patterns |
| **Cluster Compatibility** | ⚠️ Requires pandas fixes | ✅ No pandas dependency |
| **Data Fidelity** | ✅ True temporal correlations | ⚠️ Approximated correlations |
| **Reproducibility** | ✅ Consistent with HPO | ⚠️ Different data source |

## Next Steps

1. **Test**: `qsub cluster/test_real_data.pbs`
2. **Monitor**: Check test job output for pandas/data loading success
3. **Submit**: `qsub cluster/submit_real_data_fixed.pbs` 
4. **Monitor**: Track array job progress with `qstat -t`

The real data approach provides the most scientifically valid ablation study while maintaining compatibility with the cluster environment through careful pandas handling. 