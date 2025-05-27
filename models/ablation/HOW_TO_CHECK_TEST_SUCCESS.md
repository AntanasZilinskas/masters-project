# How to Check if Ablation Test was Successful

## 1. Submit and Get Job ID
```bash
cd models/ablation
job_id=$(qsub cluster/test_real_data.pbs)
echo "Submitted job: $job_id"
```

## 2. Monitor Job Progress
```bash
# Check job status
qstat -u $USER

# Watch job in real-time (optional)
watch -n 30 'qstat -u $USER'
```

## 3. Check Output Files

### Find Output Files
```bash
# List recent output files
ls -lt *.out *.err | head -5

# Or find by job ID (if you know it)
ls test_real_data.o* test_real_data.e*
```

### View Output While Running
```bash
# Watch output in real-time (replace with actual filename)
tail -f test_real_data.o12345

# Or check last 50 lines
tail -50 test_real_data.o12345
```

## 4. Success Indicators

### ✅ **SUCCESSFUL TEST OUTPUT**
Look for these key success messages in the `.out` file:

```
🧪 Testing Real Data Loading
Working directory: /rds/general/user/username/home/masters-project
Contents: cluster  models  data  ...
conda activate everest_env
Test job: Validating ablation setup
Using GPU: 0
Conda environment: everest_env
Python executable: /rds/general/user/username/home/miniforge3/envs/everest_env/bin/python
Python version: Python 3.11.x
Testing imports...
PyTorch version: 2.7.0+cu126
Ablation imports successful
✅ GPU available: NVIDIA RTX 6000
Project root: /rds/general/user/username/home/masters-project
✅ Successfully imported data functions
Loading training data...
✅ Training data: XXXX samples, shape (XXXX, 10, 9)
Loading testing data...
✅ Testing data: XXXX samples, shape (XXXX, 10, 9)
Training X shape: (XXXX, 10, 9), dtype: float64
Training y shape: (XXXX,), dtype: int64
Positive class ratio: 0.XXXX
✅ All tests passed - real data loading works!
🎯 Testing quick ablation run...
🏁 Test completed at [timestamp]
```

### ❌ **FAILURE INDICATORS**
Watch out for these error patterns:

```
❌ GPU not available - ablation cannot proceed
❌ Import failed: ModuleNotFoundError
❌ Data loading failed: FileNotFoundError
❌ Pandas completely failed
❌ Ablation test failed
```

## 5. Quick Success Check Commands

### One-liner to check if test passed
```bash
# Check if test completed successfully
tail -10 test_real_data.o* | grep -E "(✅|🏁|completed)" && echo "TEST PASSED" || echo "CHECK OUTPUT"
```

### Check for any errors
```bash
# Look for error indicators
grep -E "(❌|ERROR|Failed|failed)" test_real_data.o* test_real_data.e*
```

### Check job completion status
```bash
# If job is no longer in qstat, check exit code
qstat -x | grep test_real_data  # Shows completed jobs with exit codes
```

## 6. What Each Test Validates

| Test Component | What It Checks | Success Indicator |
|----------------|----------------|-------------------|
| **Environment** | Conda activation, Python path | `conda activate everest_env` |
| **GPU** | CUDA availability | `✅ GPU available: NVIDIA RTX 6000` |
| **Imports** | PyTorch, ablation modules | `PyTorch version: 2.7.0+cu126` |
| **Data Loading** | Real SHARP data access | `✅ Training data: XXXX samples` |
| **Ablation Run** | Quick training test | Job completes without errors |

## 7. Troubleshooting Common Issues

### Job Stuck in Queue (Status: Q)
```bash
# Check queue status
qstat -Q

# Check your job details
qstat -f JOB_ID
```

### Job Failed Immediately
```bash
# Check error file for details
cat test_real_data.e*

# Common issues:
# - Wrong working directory
# - Conda environment not found
# - GPU not available
# - Data files missing
```

### Import Errors
```bash
# Check if you're in the right directory
pwd  # Should be in masters-project/models/ablation

# Check if models directory exists
ls ../../models/
```

## 8. Next Steps After Successful Test

### If Test Passes ✅
```bash
# Submit the full ablation study
qsub cluster/submit_real_data_fixed.pbs

# Monitor the array job
qstat -u $USER
```

### If Test Fails ❌
1. **Check the error messages** in output files
2. **Verify you're in the right directory**: `masters-project/models/ablation`
3. **Check data availability**: Ensure SHARP data files exist
4. **Verify environment**: Test conda activation manually
5. **Ask for help**: Share the error output

## 9. Expected Timeline

- **Queue time**: 1-10 minutes (depending on cluster load)
- **Execution time**: 2-5 minutes for the test
- **Total time**: Usually under 15 minutes

## 10. Example Successful Session

```bash
$ cd models/ablation
$ qsub cluster/test_real_data.pbs
12345.cx3-login-1.cx3.hpc.imperial.ac.uk

$ qstat -u $USER
Job ID          Name             User              Time Use S Queue
--------------- ---------------- ----------------- -------- - -----
12345.cx3       test_real_data   username          00:00:30 R gpu24

# Wait a few minutes...

$ qstat -u $USER
# (No output - job completed)

$ ls -lt *.out | head -1
-rw-r--r-- 1 username group 2847 Dec 10 14:30 test_real_data.o12345

$ tail -5 test_real_data.o12345
✅ All tests passed - real data loading works!
🎯 Testing quick ablation run...
✅ Experiment 1/60: full_model (seed 0)
🏁 Test completed at Mon Dec 10 14:30:15 GMT 2024

$ echo "SUCCESS! Ready to submit full study"
```

This indicates everything is working and you can proceed with the full ablation study. 