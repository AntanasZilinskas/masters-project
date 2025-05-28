# How to Check if Component Ablation Test was Successful

## 🎯 **Quick Answer: What to Expect**

When you submit a test job, the cluster will create **output files** in your submission directory that contain all the logs, success/failure messages, and any errors.

## 1. Submit Test and Get Job ID
```bash
cd models/ablation
job_id=$(qsub cluster/test_component_ablation.pbs)
echo "Submitted test job: $job_id"
```

## 2. Monitor Job Progress
```bash
# Check job status
qstat -u $USER

# Job states:
# Q = Queued (waiting for resources)
# R = Running 
# C = Completed
# X = Failed/Terminated

# Watch job in real-time (optional)
watch -n 30 'qstat -u $USER'
```

## 3. 📁 **Cluster Output Files** (This is where everything goes!)

### **Automatic File Creation**
The cluster automatically creates these files in your submission directory:

```bash
# Output file (contains all stdout/print statements)
test_component_ablation.o[JOB_ID]    # e.g., test_component_ablation.o12345

# Error file (contains stderr/error messages) 
test_component_ablation.e[JOB_ID]    # e.g., test_component_ablation.e12345
```

### **Find Your Output Files**
```bash
# List recent output files (most recent first)
ls -lt test_component_ablation.o* test_component_ablation.e*

# Or find all output files
ls test_component_ablation.*
```

### **View Output While Running**
```bash
# Watch output in real-time (replace 12345 with your job ID)
tail -f test_component_ablation.o12345

# Or check last 50 lines
tail -50 test_component_ablation.o12345

# Check for any errors
cat test_component_ablation.e12345
```

## 4. ✅ **SUCCESS INDICATORS**

### **What a SUCCESSFUL test looks like:**
Look for these key messages in the `.o` (output) file:

```
🧪 Testing Component Ablation Study
Node: cx3-4-15.cx3.hpc.imperial.ac.uk
Date: Mon Dec 10 14:25:30 GMT 2024
Working directory: /rds/general/user/username/home/masters-project/models/ablation
conda activate everest_env
Test job: Validating component ablation study
Using GPU: 0
Conda environment: everest_env
Python executable: /rds/general/user/username/home/miniforge3/envs/everest_env/bin/python
Python version: Python 3.11.x

Testing imports...
PyTorch version: 2.7.0+cu126
Component ablation imports successful

Validating GPU...
✅ GPU available: NVIDIA RTX 6000

Testing data loading...
Project root: /rds/general/user/username/home/masters-project
✅ Successfully imported data functions
Loading training data...
✅ Training data: 15234 samples, shape (10, 9)
Loading testing data...
✅ Testing data: 3456 samples, shape (10, 9)
Training X shape: (15234, 10, 9), dtype: float64
Training y shape: (15234,), dtype: int64
Positive class ratio: 0.0234
✅ Data loading tests passed!

🎯 Testing component ablation...
🔬 Running component ablation: full_model, seed 0
   Input shape: (10, 9)
Loading data for M5-class, 72h window...
Data loaded successfully:
  Training: 15234 samples, shape: (15234, 10, 9)
  Validation: 3456 samples, shape: (3456, 10, 9)
  Positive rate (train): 0.023
  Positive rate (val): 0.025
Model created with ablation config:
  Attention: True
  Evidential: True
  EVT: True
  Precursor: True
Training for 50 epochs with early stopping...
✅ Experiment completed successfully!
   • Accuracy: 0.9876
   • TSS: 0.4567
   • F1: 0.1234
   • Model saved to: models/EVEREST-v1.0-M5-72h
✅ Component ablation test passed!

🎯 Testing different component variants...
  Testing no_evidential variant...
✅ no_evidential config: evidential=False, evt=True
  Testing mean_pool variant...
✅ mean_pool config: attention=False, evidential=True
✅ All component variant tests passed!

📊 Component Ablation Study Configuration:
  • Component ablations: 7 variants × 5 seeds = 35 experiments
  • Total experiments: 35

  Component variants: full_model, no_evidential, no_evt, mean_pool, cross_entropy, no_precursor, fp32_training
  Seeds: 0, 1, 2, 3, 4

✅ All tests passed - component ablation study ready!
🏁 Test completed at Mon Dec 10 14:30:15 GMT 2024
```

### **Key Success Markers:**
- ✅ GPU available
- ✅ Data loading tests passed
- ✅ Component ablation test passed
- ✅ All tests passed
- 🏁 Test completed

## 5. ❌ **FAILURE INDICATORS**

### **What FAILURE looks like:**
Watch for these error patterns in output files:

```
❌ GPU not available - ablation cannot proceed
❌ Import failed: ModuleNotFoundError: No module named 'models'
❌ Training data not found
❌ Testing data not found
❌ Data loading test failed
❌ Component ablation test failed
❌ GPU validation failed
```

### **Common Error Locations:**
- **`.e` file**: Contains Python tracebacks and system errors
- **`.o` file**: Contains script output with ❌ error markers

## 6. 🔍 **Quick Success Check Commands**

### **One-liner to check if test passed:**
```bash
# Check if test completed successfully
tail -10 test_component_ablation.o* | grep -E "(✅.*passed|🏁.*completed)" && echo "✅ TEST PASSED" || echo "❌ CHECK OUTPUT"
```

### **Check for any errors:**
```bash
# Look for error indicators in both output and error files
grep -E "(❌|ERROR|Failed|failed|Exception)" test_component_ablation.o* test_component_ablation.e*
```

### **Check job completion status:**
```bash
# If job is no longer in qstat, check exit code
qstat -x | grep test_component_ablation  # Shows completed jobs with exit codes
```

### **View the most important parts:**
```bash
# See the beginning (environment setup)
head -20 test_component_ablation.o*

# See the end (final results)
tail -20 test_component_ablation.o*

# See any errors
cat test_component_ablation.e*
```

## 7. 📊 **What Each Test Validates**

| Test Component | What It Checks | Success Indicator |
|----------------|----------------|-------------------|
| **Environment** | Conda activation, Python path | `conda activate everest_env` |
| **GPU** | CUDA availability | `✅ GPU available: NVIDIA RTX 6000` |
| **Imports** | PyTorch, ablation modules | `Component ablation imports successful` |
| **Data Loading** | Real SHARP data access | `✅ Training data: XXXX samples` |
| **Component Test** | Quick ablation run | `✅ Component ablation test passed!` |
| **Variant Configs** | Different ablation setups | `✅ All component variant tests passed!` |

## 8. 🚨 **Troubleshooting Common Issues**

### **Job Stuck in Queue (Status: Q)**
```bash
# Check queue status and available resources
qstat -Q
showq  # Shows cluster load

# Check your job details
qstat -f JOB_ID
```

### **Job Failed Immediately (Status: X)**
```bash
# Check error file for details
cat test_component_ablation.e*

# Common issues:
# - Wrong working directory
# - Conda environment not found  
# - GPU not available
# - Data files missing
# - Import errors
```

### **Import/Module Errors**
```bash
# Verify you're in the right directory
pwd  # Should end with: masters-project/models/ablation

# Check if models directory exists
ls ../../models/solarknowledge_ret_plus.py
ls ../../models/utils.py
```

### **Data Loading Errors**
```bash
# Check if data directory exists
ls ../../data/

# Test data loading manually
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent.parent))
from models.utils import get_training_data
X, y = get_training_data('72', 'M5')
print(f'Data loaded: {len(X) if X else 0} samples')
"
```

## 9. ⏱️ **Expected Timeline**

- **Queue time**: 1-10 minutes (depending on cluster load)
- **Execution time**: 3-8 minutes for the test
- **Total time**: Usually under 15 minutes

## 10. 🎯 **Next Steps After Test**

### **If Test Passes ✅**
```bash
# Submit the full component ablation study (35 experiments)
qsub cluster/submit_component_ablation.pbs

# Monitor the array job
qstat -u $USER
```

### **If Test Fails ❌**
1. **Check the error messages** in `.o` and `.e` files
2. **Verify directory**: Should be in `masters-project/models/ablation`
3. **Check data availability**: Ensure SHARP data files exist
4. **Verify environment**: Test conda activation manually
5. **Share error output**: Copy the error messages for help

## 11. 📝 **Example Successful Session**

```bash
$ cd models/ablation
$ qsub cluster/test_component_ablation.pbs
12345.cx3-login-1.cx3.hpc.imperial.ac.uk

$ qstat -u $USER
Job ID          Name                    User        Time Use S Queue
--------------- ----------------------- ----------- -------- - -----
12345.cx3       test_component_ablation username    00:00:30 R gpu24

# Wait 5-10 minutes...

$ qstat -u $USER
# (No output - job completed)

$ ls -lt test_component_ablation.*
-rw-r--r-- 1 username group 4521 Dec 10 14:30 test_component_ablation.o12345
-rw-r--r-- 1 username group    0 Dec 10 14:30 test_component_ablation.e12345

$ tail -5 test_component_ablation.o12345
✅ All component variant tests passed!
✅ All tests passed - component ablation study ready!
🏁 Test completed at Mon Dec 10 14:30:15 GMT 2024

$ echo "SUCCESS! Ready to submit full component ablation study"
```

## 12. 🔄 **Monitoring the Full Study**

Once the test passes and you submit the full study:

```bash
# Submit full study
qsub cluster/submit_component_ablation.pbs

# Monitor all 35 jobs
qstat -u $USER

# Check progress of specific jobs
tail -f everest_component_ablation.o12346  # Job 1
tail -f everest_component_ablation.o12347  # Job 2
# etc.
```

**Remember**: The cluster creates separate output files for each job in the array, so you'll have 35 different output files when running the full study! 