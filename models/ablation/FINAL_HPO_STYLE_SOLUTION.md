# EVEREST Ablation Study - Final HPO-Style Solution

## Problem Summary

The ablation study was failing on Imperial RCS cluster due to **import path and execution pattern differences** from the working HPO study. After detailed analysis, I identified the exact differences and created a solution that follows the **identical pattern** as the successful HPO.

## Root Cause Analysis

### Working HPO Pattern:
1. **Path Setup**: `sys.path.insert(0, str(project_root))` + `os.chdir(project_root)`
2. **Direct Script Execution**: `python models/hpo/run_hpo.py`
3. **Data Validation**: Validates data availability before starting
4. **GPU Validation**: Validates GPU before proceeding
5. **Error Handling**: Comprehensive validation and error reporting

### Previous Ablation Issues:
1. **Module Import Approach**: Used `python -m ablation.trainer` (failed)
2. **Relative Import Issues**: `from .config import` only works in package context
3. **Missing Validations**: No data/GPU validation like HPO
4. **Different Execution Pattern**: Different from proven HPO approach

## Final Solution

### 1. HPO-Style Runner: `run_ablation_hpo_style.py`

**Key Features:**
- **Identical path setup** as HPO: `sys.path.insert(0, str(project_root))` + `os.chdir(project_root)`
- **Data validation** before training (exact same as HPO)
- **GPU validation** before training (exact same as HPO)
- **Direct script execution** pattern (exact same as HPO)
- **Comprehensive error handling** and reporting

**Usage:**
```bash
python models/ablation/run_ablation_hpo_style.py --variant full_model --seed 0
python models/ablation/run_ablation_hpo_style.py --variant no_evidential --seed 1
python models/ablation/run_ablation_hpo_style.py --variant full_model --seed 0 --sequence seq_15
```

### 2. HPO-Style Cluster Script: `submit_exact_hpo_style.pbs`

**Key Features:**
- **Identical PBS directives** as working HPO
- **Identical environment setup** as working HPO
- **Identical validation steps** as working HPO
- **Same resource monitoring** as working HPO
- **Same error handling** as working HPO

**Experiment Distribution:**
- **60 total experiments**: 35 component ablations + 25 sequence ablations
- **10 array jobs**: 6 experiments each
- **Component ablations**: 7 variants × 5 seeds = 35 experiments
- **Sequence ablations**: 5 variants × 5 seeds = 25 experiments

## Submission Instructions

### Step 1: Cancel Current Jobs (if needed)
```bash
# Check current jobs
qstat -u $USER -t

# Cancel if needed
qdel 1168903
```

### Step 2: Submit New HPO-Style Array Job
```bash
cd /rds/general/user/az2221/home/masters-project
qsub models/ablation/cluster/submit_exact_hpo_style.pbs
```

### Step 3: Monitor Progress
```bash
# Check job status
qstat -u $USER -t

# Check logs (once jobs start)
ls ablation_exact_*.out
tail -f ablation_exact_1.out
```

## Expected Behavior

### Successful Job Output:
```
Working directory: /rds/general/user/az2221/home/masters-project
Conda environment: everest_env
Testing imports...
PyTorch version: 2.x.x
Ablation imports successful
Validating GPU...
✅ GPU available: Tesla V100-SXM2-32GB
Starting ablation experiments...
🔬 [1/6] Global: [1/60] Running: component:full_model:0
🎯 Running ablation: full_model, seed 0
   • Validating data availability...
   ✅ Data validated: 400000+ train, 100000+ test samples
   • Validating GPU configuration...
   ✅ GPU available: Tesla V100-SXM2-32GB (device 0/1)
   ✅ Completed successfully!
   • TSS: 0.xxxx
   • F1: 0.xxxx
```

### Job Distribution:
- **Array Job 1**: Experiments 1-6 (component:full_model:0 through component:no_evidential:0)
- **Array Job 2**: Experiments 7-12 (component:no_evidential:1 through component:no_evt:1)
- **Array Job 3**: Experiments 13-18 (component:no_evt:2 through component:mean_pool:2)
- **Array Job 4**: Experiments 19-24 (component:mean_pool:3 through component:cross_entropy:3)
- **Array Job 5**: Experiments 25-30 (component:cross_entropy:4 through component:no_precursor:4)
- **Array Job 6**: Experiments 31-36 (component:fp32_training:0 through component:fp32_training:4, sequence:seq_5:0)
- **Array Job 7**: Experiments 37-42 (sequence:seq_5:1 through sequence:seq_7:1)
- **Array Job 8**: Experiments 43-48 (sequence:seq_7:2 through sequence:seq_10:2)
- **Array Job 9**: Experiments 49-54 (sequence:seq_10:3 through sequence:seq_15:3)
- **Array Job 10**: Experiments 55-60 (sequence:seq_15:4 through sequence:seq_20:4)

## Key Differences from Previous Attempts

| Aspect | Previous (Failed) | New HPO-Style (Should Work) |
|--------|------------------|----------------------------|
| **Execution** | `python -m ablation.trainer` | `python models/ablation/run_ablation_hpo_style.py` |
| **Path Setup** | Relative imports | `sys.path.insert(0, str(project_root))` + `os.chdir(project_root)` |
| **Validation** | Minimal | Data + GPU validation (same as HPO) |
| **Error Handling** | Basic | Comprehensive (same as HPO) |
| **Import Pattern** | `from .config import` | Direct imports after path setup |
| **Structure** | Module-based | Script-based (same as HPO) |

## Why This Should Work

1. **Proven Pattern**: Uses the **exact same pattern** as the working HPO
2. **Identical Environment**: Same conda setup, same path configuration
3. **Same Validations**: Same data and GPU validation as HPO
4. **Same Resource Requests**: Same PBS directives as working HPO
5. **Same Error Handling**: Same comprehensive error checking as HPO

## Troubleshooting

If jobs still fail, check:

1. **Log Files**: `ablation_exact_*.out` and `ablation_exact_*.err`
2. **Import Test**: 
   ```bash
   python -c "import models.ablation; print('Success')"
   ```
3. **Data Availability**:
   ```bash
   python -c "from models.utils import get_training_data; print('Data OK' if get_training_data('72', 'M5')[0] is not None else 'Data Missing')"
   ```
4. **GPU Test**:
   ```bash
   python -c "import torch; print('GPU OK' if torch.cuda.is_available() else 'No GPU')"
   ```

## Expected Timeline

- **Job Submission**: Immediate
- **Queue Time**: 5-30 minutes (depending on cluster load)
- **Execution Time**: ~2-3 hours per array job (6 experiments × 20-30 min each)
- **Total Time**: 2-3 hours for all 60 experiments (parallel execution)

## Results Location

Results will be saved to:
- **Experiment Results**: `models/ablation/results/`
- **Model Weights**: `models/ablation/trained_models/`
- **Logs**: `models/ablation/logs/`
- **Job Logs**: `ablation_exact_*.out` files in submission directory

This solution follows the **exact same pattern** as the working HPO and should resolve all previous import and execution issues. 