# EVEREST Ablation Study - Ultra Robust Solution

## Current Situation Analysis

You're experiencing **partial failures** in the cluster array jobs:
- Jobs 1 & 2: Failed (status X)
- Jobs 3 & 4: Running (status R) 
- Jobs 5-10: Queued (status Q)

This pattern suggests **timing or resource contention issues** rather than fundamental code problems, since some jobs are successfully running.

## Root Cause Hypothesis

The failures are likely due to:
1. **Race conditions** during conda environment activation
2. **Resource contention** when multiple jobs start simultaneously
3. **Temporary file system issues** during job initialization
4. **GPU allocation conflicts** between concurrent jobs

## Ultra Robust Solution

I've created two new scripts with comprehensive error handling:

### 1. Single Test Job: `test_single_experiment.pbs`
**Purpose**: Validate the setup with one experiment before running the full array
**Runtime**: ~30-60 minutes
**Resources**: 1 GPU, 4 CPUs, 24GB RAM

### 2. Ultra Robust Array: `submit_ultra_robust.pbs`
**Purpose**: Run all 60 experiments with maximum error handling
**Features**:
- **Comprehensive logging** with timestamps
- **Retry logic** for conda activation
- **Step-by-step validation** with detailed error reporting
- **Timeout protection** (1 hour per experiment)
- **Resource monitoring** throughout execution
- **Graceful error handling** and recovery

## Recommended Submission Strategy

### Step 1: Cancel Current Jobs
```bash
# Check current status
qstat -u $USER -t

# Cancel the failing array job
qdel 1168922
```

### Step 2: Test Single Experiment First
```bash
cd /rds/general/user/az2221/home/masters-project
qsub models/ablation/cluster/test_single_experiment.pbs
```

**Wait for this to complete successfully before proceeding!**

### Step 3: Monitor Test Job
```bash
# Check status
qstat -u $USER

# Once completed, check logs
cat ablation_test_single.out
cat ablation_test_single.err
```

**Expected successful output:**
```
[2024-XX-XX XX:XX:XX] === EVEREST Ablation Study - Single Experiment Test ===
[2024-XX-XX XX:XX:XX] Initializing conda...
[2024-XX-XX XX:XX:XX] Activating conda environment...
[2024-XX-XX XX:XX:XX] Conda environment: everest_env
[2024-XX-XX XX:XX:XX] Testing imports...
PyTorch: 2.x.x
Ablation imports: OK
[2024-XX-XX XX:XX:XX] GPU validation...
GPU: Tesla V100-SXM2-32GB
[2024-XX-XX XX:XX:XX] Testing data availability...
Data: 709447 train, 71729 test samples
[2024-XX-XX XX:XX:XX] Running single test experiment...
🔬 EVEREST Ablation Study Framework
🎯 Running ablation: full_model, seed 0
   ✅ Data validated: 709447 train, 71729 test samples
   ✅ GPU available: Tesla V100-SXM2-32GB
   ✅ Completed successfully!
[2024-XX-XX XX:XX:XX] Test completed successfully!
```

### Step 4: Submit Ultra Robust Array (Only if Step 2 succeeds)
```bash
qsub models/ablation/cluster/submit_ultra_robust.pbs
```

## Ultra Robust Features

### Enhanced Error Handling
- **Set -e**: Exit immediately on any error
- **Set -u**: Exit on undefined variables
- **Error logging**: All errors logged with timestamps
- **Validation steps**: Each step validated before proceeding

### Retry Logic
- **Conda activation**: 3 attempts with 5-second delays
- **Environment verification**: Comprehensive checks
- **GPU validation**: Memory allocation test

### Comprehensive Logging
```bash
# Each job creates detailed logs
ablation_ultra_1.out  # Stdout with timestamps
ablation_ultra_1.err  # Stderr with error details
```

### Resource Monitoring
- **Memory usage**: Tracked throughout execution
- **GPU memory**: Monitored before and during experiments
- **Disk space**: Verified sufficient space available
- **Timing**: Each experiment timed individually

### Experiment Protection
- **Timeout**: 1 hour maximum per experiment
- **Error isolation**: One failed experiment doesn't stop others
- **Progress tracking**: Success/failure counts maintained
- **Resource cleanup**: Brief pauses between experiments

## Troubleshooting Guide

### If Single Test Fails:

1. **Check conda environment**:
   ```bash
   ssh to cluster
   source ~/miniforge3/etc/profile.d/conda.sh
   conda activate everest_env
   python -c "import models.ablation; print('OK')"
   ```

2. **Check data availability**:
   ```bash
   python -c "from models.utils import get_training_data; print('OK' if get_training_data('72', 'M5')[0] is not None else 'FAIL')"
   ```

3. **Check GPU access**:
   ```bash
   python -c "import torch; print('OK' if torch.cuda.is_available() else 'FAIL')"
   ```

### If Array Jobs Partially Fail:

1. **Check logs** for specific error patterns:
   ```bash
   grep "ERROR" ablation_ultra_*.out
   grep "Failed" ablation_ultra_*.out
   ```

2. **Identify successful vs failed jobs**:
   ```bash
   grep "All experiments completed successfully" ablation_ultra_*.out
   ```

3. **Check resource contention**:
   ```bash
   grep "Memory info" ablation_ultra_*.out
   grep "GPU memory" ablation_ultra_*.out
   ```

## Expected Timeline

- **Single Test**: 30-60 minutes
- **Array Job Queue Time**: 5-30 minutes
- **Array Job Execution**: 2-3 hours (parallel)
- **Total Time**: 3-4 hours from start to finish

## Success Indicators

### Single Test Success:
- ✅ Conda environment activated
- ✅ All imports successful
- ✅ GPU detected and accessible
- ✅ Data loaded successfully
- ✅ One experiment completed with TSS/F1 scores

### Array Job Success:
- ✅ All 10 array jobs start successfully
- ✅ No jobs fail with status X
- ✅ Each job completes 6 experiments
- ✅ 60 total experiments completed
- ✅ Results saved to `models/ablation/results/`

## Next Steps After Success

Once all experiments complete successfully:

1. **Collect Results**:
   ```bash
   ls models/ablation/results/
   ```

2. **Run Analysis**:
   ```bash
   python models/ablation/analysis.py
   ```

3. **Generate Plots**:
   ```bash
   python models/ablation/create_plots.py
   ```

This ultra-robust approach should resolve the intermittent failures you're experiencing by adding comprehensive error handling, retry logic, and detailed logging to identify and resolve any remaining issues. 