# EVEREST Ablation Study - Library Compatibility Fix

## 🔍 **Problem Identified**

The ablation study was failing due to a **libstdc++ compatibility issue** with pandas:

```
ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
```

This occurs when the conda environment has newer C++ libraries than the cluster system.

## 🛠️ **Solution Overview**

The fix involves:

1. **Loading system modules** for GCC/CUDA compatibility
2. **Setting LD_LIBRARY_PATH** to prioritize conda libraries
3. **Graceful pandas handling** with fallback options
4. **Enhanced error detection** and recovery

## 📋 **Fixed Scripts**

### 1. Test Script: `test_single_fixed.pbs`
- Single experiment test with compatibility fixes
- Runtime: 1 hour
- Comprehensive validation steps

### 2. Production Script: `submit_ultra_robust_fixed.pbs`
- Full array job (10 jobs × 6 experiments each)
- Runtime: 12 hours per job
- Enhanced error handling and recovery

## 🚀 **Submission Instructions**

### Step 1: Test the Fix
```bash
cd /rds/general/user/az2221/home/repositories/masters-project/models/ablation
qsub cluster/test_single_fixed.pbs
```

### Step 2: Monitor Test Job
```bash
# Check job status
qstat -u az2221

# Monitor output (replace JOBID with actual job ID)
tail -f ablation_test_fixed.out
```

### Step 3: Verify Test Success
The test should show:
```
✅ NumPy: [version]
✅ PyTorch: [version]
✅ CUDA available: [GPU name]
✅ Pandas: [version] OR ⚠️ Pandas has libstdc++ compatibility issue - continuing without pandas
✅ Ablation imports successful
✅ All tests passed - ready for full training
```

### Step 4: Submit Full Array Job (Only After Test Passes)
```bash
qsub cluster/submit_ultra_robust_fixed.pbs
```

## 🔧 **Key Fixes Applied**

### 1. System Module Loading
```bash
module load gcc/9.3.0 2>/dev/null || log "GCC module not available"
module load cuda/11.8 2>/dev/null || log "CUDA module not available"
```

### 2. Library Path Priority
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

### 3. Pandas Compatibility Handling
```python
try:
    import pandas as pd
    print(f'✅ Pandas: {pd.__version__}')
except ImportError as e:
    if 'GLIBCXX' in str(e):
        print('⚠️ Pandas has libstdc++ compatibility issue - continuing without pandas')
    else:
        print(f'❌ Pandas failed: {e}')
        sys.exit(1)
```

### 4. Enhanced Error Recovery
- Retry logic for conda activation
- Graceful handling of library issues
- Detailed error reporting
- Timeout protection for experiments

## 📊 **Expected Timeline**

### Test Job (1 hour)
- Environment setup: 5-10 minutes
- Validation steps: 5-10 minutes
- Single experiment test: 30-40 minutes

### Full Array Job (12 hours per job)
- 10 parallel jobs
- 6 experiments per job (60 total)
- Each experiment: 1-2 hours
- Total completion: 12-15 hours

## 🎯 **Success Criteria**

### Test Job Success
- All validation steps pass ✅
- No library compatibility errors
- Single experiment completes successfully

### Production Job Success
- All 10 array jobs complete
- 60 experiments total (35 component + 25 sequence)
- Results saved in `models/ablation/results/`
- Trained models saved in `models/ablation/trained_models/`

## 🔍 **Monitoring Commands**

```bash
# Check all jobs
qstat -u az2221

# Monitor specific job output
tail -f ablation_ultra_fixed_1.out

# Check job details
qstat -f JOBID

# Monitor GPU usage
ssh NODE_NAME nvidia-smi
```

## 🚨 **Troubleshooting**

### If Test Job Still Fails
1. Check the error in `ablation_test_fixed.err`
2. Verify conda environment exists: `conda env list`
3. Check available modules: `module avail`

### If Array Jobs Fail
1. Check individual job outputs: `ablation_ultra_fixed_X.out`
2. Look for specific error patterns
3. Consider reducing batch size or timeout

### Common Issues
- **Queue limits**: Wait for slots to become available
- **Memory issues**: Jobs may need more than 24GB for large models
- **Timeout**: Some experiments may need more than 1 hour

## 📈 **Expected Results**

After successful completion, you should have:

```
models/ablation/results/
├── component_ablation_results.csv
├── sequence_ablation_results.csv
└── ablation_summary.json

models/ablation/trained_models/
├── full_model_seed_0/
├── no_evidential_seed_0/
├── seq_5_seed_0/
└── ... (60 total model directories)
```

## 🎉 **Next Steps After Completion**

1. **Analyze Results**: Run `python models/ablation/analysis.py`
2. **Generate Plots**: Results will include performance comparisons
3. **Write Thesis**: Use results for ablation study chapter

---

**Note**: This fix addresses the specific libstdc++ compatibility issue while maintaining all the robustness features of the original ultra-robust solution. 