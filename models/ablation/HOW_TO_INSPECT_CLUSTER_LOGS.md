# How to Inspect Cluster Job Logs on Imperial RCS

## 1. Understanding PBS Job Log Files

When you submit a PBS job, the system creates two types of log files:
- **Output file**: `jobname.o[JOBID]` - Contains stdout (normal output)
- **Error file**: `jobname.e[JOBID]` - Contains stderr (error messages)

## 2. Finding Your Job Logs

### Method 1: Check Submission Directory
```bash
# Navigate to where you submitted the job
cd models/ablation/cluster

# Look for .o and .e files
ls -la *.o* *.e* 2>/dev/null

# Or search more broadly
find . -name "*.o*" -o -name "*.e*" | head -10
```

### Method 2: Check Your Home Directory
```bash
# PBS sometimes puts logs in your home directory
cd ~
ls -la *.o* *.e* 2>/dev/null
```

### Method 3: Check Current Working Directory
```bash
# Check where you are when logs might be created
pwd
ls -la *.o* *.e* 2>/dev/null
```

## 3. Checking Job Status First

### Check Current Jobs
```bash
# See your running/queued jobs
qstat -u $USER

# More detailed view
qstat -f -u $USER

# Check specific job
qstat -f [JOBID]
```

### Check Recent Job History
```bash
# Show completed jobs (if available)
qstat -x -u $USER | tail -20

# Or check job accounting (if available)
tracejob [JOBID]
```

## 4. Real-Time Log Monitoring

### For Running Jobs
```bash
# Monitor output in real-time (if file exists)
tail -f jobname.o[JOBID]

# Monitor both output and error
tail -f jobname.o[JOBID] &
tail -f jobname.e[JOBID] &
```

### For Array Jobs
```bash
# Array jobs create multiple log files
ls -la *.o[JOBID].*  # Output files for each array element
ls -la *.e[JOBID].*  # Error files for each array element

# Check specific array element
cat jobname.o[JOBID].[ARRAY_INDEX]
```

## 5. Common Log File Locations

### Based on Your Submission
If you submitted from `models/ablation/cluster/`, logs might be in:
1. `models/ablation/cluster/` (submission directory)
2. `models/ablation/` (parent directory)
3. `~/` (home directory)
4. `/tmp/` (temporary directory - rare)

### Search Command
```bash
# Search for recent log files across common locations
find ~ models/ablation -name "*.o*" -o -name "*.e*" -newer /tmp/timestamp 2>/dev/null
```

## 6. Analyzing Log Contents

### Check for Common Issues
```bash
# Look for error patterns
grep -i "error\|failed\|exception\|traceback" *.e* *.o*

# Check for memory issues
grep -i "memory\|oom\|killed" *.e* *.o*

# Check for GPU issues
grep -i "cuda\|gpu\|device" *.e* *.o*

# Check for module/import issues
grep -i "module\|import\|not found" *.e* *.o*
```

### Check Job Resource Usage
```bash
# Look for resource information in logs
grep -i "resource\|time\|memory\|cpu" *.o*

# Check if job completed
grep -i "complete\|finished\|done" *.o*
```

## 7. Debugging Specific Issues

### Script Not Found
```bash
# Check if the script path is correct
ls -la /path/to/your/script.py

# Check permissions
ls -la models/ablation/run_ablation_*.py
```

### Environment Issues
```bash
# Check if conda environment exists
conda env list | grep everest

# Check if modules are available
module avail | grep -i python
```

### Data Loading Issues
```bash
# Check if data files exist
ls -la data/
ls -la models/ablation/data/
```

## 8. Emergency Log Recovery

### If No Log Files Found
```bash
# Check system logs (if accessible)
journalctl -u pbs

# Check scheduler logs
ls -la /var/spool/pbs/server_logs/

# Contact system administrators
# Email: rcs-support@imperial.ac.uk
```

## 9. Your Current Situation

Based on your setup, try these commands:

```bash
# 1. Check for recent job logs
cd models/ablation/cluster
ls -la *.o* *.e* 2>/dev/null

# 2. Check parent directory
cd ..
ls -la *.o* *.e* 2>/dev/null

# 3. Check home directory
cd ~
ls -la *.o* *.e* 2>/dev/null

# 4. Search broadly for recent logs
find ~ -name "*.o*" -o -name "*.e*" -newer $(date -d "1 day ago" +%Y%m%d) 2>/dev/null

# 5. Check job status
qstat -u $USER

# 6. Check recent job history
qstat -x -u $USER | tail -10
```

## 10. Alternative Debugging Methods

### Test Script Locally First
```bash
# Test the script locally to isolate cluster vs code issues
cd models/ablation
python run_ablation_no_pandas.py --variant full_model --seed 0
```

### Use Interactive Session
```bash
# Request interactive session for debugging
qsub -I -l select=1:ncpus=4:mem=16gb:ngpus=1 -l walltime=1:00:00
```

### Enable Verbose Logging
Add these to your PBS script:
```bash
set -x  # Enable command tracing
set -e  # Exit on any error
```

## 11. Contact Information

If logs are still not found or accessible:
- **RCS Support**: rcs-support@imperial.ac.uk
- **Include**: Job ID, submission time, script name, error description 