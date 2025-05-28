# Quick Commands for Cluster Log Inspection

## When you're logged into Imperial RCS cluster, run these commands:

### 1. Check Current Job Status
```bash
# See your running/queued jobs
qstat -u $USER

# More detailed view of all your jobs
qstat -f -u $USER

# Check recent job history (completed jobs)
qstat -x -u $USER | tail -20
```

### 2. Find Log Files in Common Locations
```bash
# Check submission directory
cd models/ablation/cluster
ls -la *.o* *.e* 2>/dev/null

# Check parent directory
cd ../
ls -la *.o* *.e* 2>/dev/null

# Check home directory
cd ~
ls -la *.o* *.e* 2>/dev/null

# Search broadly for recent logs
find ~ -name "*.o*" -o -name "*.e*" -mtime -7 2>/dev/null
```

### 3. Analyze Specific Log Files
```bash
# View the most recent output file
ls -t *.o* | head -1 | xargs cat

# View the most recent error file
ls -t *.e* | head -1 | xargs cat

# Search for errors in all log files
grep -i "error\|failed\|exception" *.o* *.e* 2>/dev/null

# Check for specific issues
grep -i "pandas\|module\|import" *.o* *.e* 2>/dev/null
grep -i "memory\|oom\|killed" *.o* *.e* 2>/dev/null
grep -i "cuda\|gpu" *.o* *.e* 2>/dev/null
```

### 4. Monitor Running Jobs in Real-Time
```bash
# If you know the job ID, monitor output
tail -f jobname.o[JOBID]

# For array jobs, check specific array element
tail -f jobname.o[JOBID].[ARRAY_INDEX]
```

### 5. Quick Diagnostic Commands
```bash
# Check if your script exists and is executable
ls -la models/ablation/run_ablation_*.py

# Check if conda environment is available
conda env list | grep everest

# Check available modules
module avail | grep -i python

# Check disk space (in case of storage issues)
df -h $HOME
```

## Most Likely Log File Names for Your Jobs:

Based on your PBS scripts, look for files like:
- `test_component_no_pandas.o[JOBID]`
- `test_component_no_pandas.e[JOBID]`
- `submit_component_no_pandas.o[JOBID]`
- `submit_component_no_pandas.e[JOBID]`

## If No Logs Found:

1. **Check job was actually submitted:**
   ```bash
   qstat -x -u $USER | grep -E "(component|ablation)"
   ```

2. **Check for permission issues:**
   ```bash
   ls -la models/ablation/cluster/
   ```

3. **Check if job is still queued:**
   ```bash
   qstat -u $USER
   ```

4. **Use interactive session for debugging:**
   ```bash
   qsub -I -l select=1:ncpus=4:mem=16gb:ngpus=1 -l walltime=1:00:00
   ```

## Emergency Contact:
If logs are completely missing or inaccessible:
- Email: rcs-support@imperial.ac.uk
- Include: Your username, job submission time, script name 