# How to Check Production Training Logs

## 🎯 **Quick Answer: Where to Find Your Logs**

When you submit production training jobs, the cluster creates **output files** that contain all logs, success/failure messages, and errors.

## 1. Check Job Status First

```bash
# Check current job status
qstat -u $USER

# Job states:
# Q = Queued (waiting for resources)
# R = Running 
# C = Completed
# X = Failed/Terminated
# E = Exiting

# For array jobs, see all sub-jobs
qstat -t [JOB_ID]  # Replace [JOB_ID] with your actual job ID
```

## 2. 📁 **Find Your Log Files**

### **Automatic Log File Creation**
The cluster creates these files in your submission directory (`models/training/cluster/`):

```bash
# For array jobs (45 experiments):
production_training.o[JOB_ID].[ARRAY_INDEX]    # e.g., production_training.o12345.1
production_training.e[JOB_ID].[ARRAY_INDEX]    # e.g., production_training.e12345.1

# For test jobs:
test_production_training.o[JOB_ID]              # e.g., test_production_training.o12345
test_production_training.e[JOB_ID]              # e.g., test_production_training.e12345
```

### **List Your Log Files**
```bash
cd models/training/cluster

# List all production training logs (most recent first)
ls -lt production_training.* test_production_training.*

# Count how many array jobs completed
ls production_training.o* | wc -l

# Find specific array job logs
ls production_training.*.[1-9]*  # Jobs 1-9
ls production_training.*.1[0-9]  # Jobs 10-19
ls production_training.*.2[0-9]  # Jobs 20-29
```

## 3. ✅ **Check for SUCCESS**

### **View a Specific Job Log**
```bash
# Replace 12345.1 with your actual job ID and array index
cat production_training.o12345.1

# Or view last 50 lines
tail -50 production_training.o12345.1

# Check for errors
cat production_training.e12345.1
```

### **SUCCESS Indicators to Look For:**
```
✅ Production training experiment completed successfully
   Job: 1/45
   Completed at: Mon Dec 10 15:30:45 GMT 2024

✅ Experiment completed successfully
   Name: c_24h_seed0
   TSS: 0.1234
   Accuracy: 0.8567
   F1: 0.2345
   Optimal threshold: 0.456
```

### **Quick Success Check for All Jobs**
```bash
# Count successful completions
grep -l "completed successfully" production_training.o*

# Count failures
grep -l "failed" production_training.o*

# Summary of all jobs
for file in production_training.o*; do
    if grep -q "completed successfully" "$file"; then
        echo "✅ $file: SUCCESS"
    elif grep -q "failed" "$file"; then
        echo "❌ $file: FAILED"
    else
        echo "⏳ $file: RUNNING/UNKNOWN"
    fi
done
```

## 4. 🔍 **Debugging FAILURES**

### **Common Error Patterns to Look For:**

#### **1. Environment Issues**
```bash
# Look for these error patterns:
grep -n "ModuleNotFoundError\|ImportError\|conda\|environment" production_training.e*
```

#### **2. GPU Issues**
```bash
# Check for GPU problems:
grep -n "CUDA\|GPU\|device" production_training.e*
```

#### **3. Data Loading Issues**
```bash
# Check for data problems:
grep -n "Data loading\|pandas\|GLIBCXX\|get_training_data" production_training.e*
```

#### **4. Memory Issues**
```bash
# Check for memory problems:
grep -n "memory\|OOM\|killed" production_training.e*
```

### **View Detailed Error Information**
```bash
# For a failed job, check both output and error files
failed_job="production_training.o12345.1"  # Replace with actual failed job

echo "=== OUTPUT FILE ==="
tail -100 "$failed_job"

echo "=== ERROR FILE ==="
error_file="${failed_job//.o/.e}"
cat "$error_file"
```

## 5. 📊 **Monitor Running Jobs**

### **Real-time Monitoring**
```bash
# Watch job status (updates every 30 seconds)
watch -n 30 'qstat -u $USER'

# Follow a running job's output
tail -f production_training.o12345.1  # Replace with actual job
```

### **Check Resource Usage**
```bash
# For running jobs, check resource usage
qstat -f [JOB_ID] | grep -E "(resources_used|Resource_List)"
```

## 6. 🔧 **Common Issues and Solutions**

### **Issue 1: Jobs Stuck in Queue (Q state)**
```bash
# Check queue status
qstat -Q

# Check your job limits
qstat -u $USER | wc -l  # Count your jobs

# Solution: Wait for resources or reduce resource requests
```

### **Issue 2: Import Errors**
Look for in error files:
```
ModuleNotFoundError: No module named 'models'
```
**Solution:** Environment setup issue - rerun test job first.

### **Issue 3: GPU Not Available**
Look for:
```
❌ GPU not available - training cannot proceed
```
**Solution:** GPU resource not allocated - check PBS resource requests.

### **Issue 4: Pandas Library Issues**
Look for:
```
GLIBCXX_3.4.29' not found (required by pandas)
```
**Solution:** Use the pandas-free version or fix conda environment.

## 7. 📈 **Results Location**

### **Successful Jobs Create These Directories:**
```bash
# Check for saved models (in project root)
ls models/EVEREST-v*

# Check for results files
ls models/training/results/

# Check for trained model weights
ls models/training/trained_models/
```

### **View Training Results**
```bash
# List all trained models
ls -la models/EVEREST-v*

# Check metadata for a specific model
cat models/EVEREST-v*/metadata.json | head -20

# View training history
cat models/EVEREST-v*/training_history.csv | head -10
```

## 8. 🚨 **Emergency Debugging**

### **If Everything Seems Broken:**
```bash
# 1. Check if you're in the right directory
pwd
ls models/training/cluster/

# 2. Check recent job submissions
qstat -u $USER

# 3. Look at the most recent log
ls -t production_training.* | head -1 | xargs tail -50

# 4. Check cluster status
qstat -Q
pbsnodes -a | grep -E "(state|jobs)"

# 5. Rerun test job to verify environment
qsub test_production_training.pbs
```

### **Get Help from Logs:**
```bash
# Create a summary of all job statuses
echo "=== JOB SUMMARY ==="
for i in {1..45}; do
    log_file=$(ls production_training.o*.$i 2>/dev/null | head -1)
    if [ -n "$log_file" ]; then
        if grep -q "completed successfully" "$log_file"; then
            echo "Job $i: ✅ SUCCESS"
        elif grep -q "failed" "$log_file"; then
            echo "Job $i: ❌ FAILED"
        else
            echo "Job $i: ⏳ RUNNING"
        fi
    else
        echo "Job $i: ⭕ NOT STARTED"
    fi
done
```

## 9. 📞 **When to Contact Support**

Contact cluster support if you see:
- Jobs stuck in queue for >24 hours
- Repeated node failures
- Filesystem errors
- Hardware-related GPU errors

**Include in your support request:**
- Job ID(s)
- Error messages from log files
- Resource requests used
- What you were trying to accomplish

---

## Quick Reference Commands

```bash
# Check job status
qstat -u $USER

# List log files
ls -lt production_training.*

# Check success/failure
grep -l "completed successfully" production_training.o*
grep -l "failed" production_training.o*

# View specific job log
tail -50 production_training.o[JOB_ID].[ARRAY_INDEX]

# Monitor running job
tail -f production_training.o[JOB_ID].[ARRAY_INDEX]

# Check for common errors
grep -n "Error\|Failed\|Exception" production_training.e*
``` 