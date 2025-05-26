# Running HPO Study on Imperial RCS Cluster

This directory contains scripts and documentation for running the EVEREST solar flare prediction HPO (Hyperparameter Optimization) study on the Imperial College London Research Computing Service (RCS) cluster.

## Prerequisites

1. **Access to Imperial RCS**: Ensure you have access to the CX3 cluster
2. **SSH Connection**: Connect to the cluster via SSH
3. **Data**: Ensure your solar flare datasets are available on the cluster

## Cluster Overview

The Imperial RCS provides two main clusters:
- **CX3**: General purpose cluster (what we'll use)
- **HX1**: Capability cluster for large multi-node jobs

Available GPU resources on CX3:
- **L40S**: 48GB GDDR6 (recommended for our workload)
- **A100**: 40GB GDDR6 (limited availability)
- **A40**: 48GB GDDR6 (JupyterHub only)
- **RTX6000**: 24GB GDDR6 (JupyterHub only)

## Quick Start

### 1. Connect to the Cluster

```bash
ssh username@login.cx3.hpc.imperial.ac.uk
```

### 2. Transfer Your Code

Upload your EVEREST project to the cluster:

```bash
# From your local machine
scp -r /path/to/masters-project username@login.cx3.hpc.imperial.ac.uk:~/
```

### 3. Set Up Environment and Run

```bash
# Navigate to your project
cd ~/masters-project

# Make scripts executable
chmod +x models/hpo/cluster/submit_jobs.sh

# Run complete workflow (setup + all 9 HPO targets)
./models/hpo/cluster/submit_jobs.sh all
```

## Detailed Usage

### Environment Setup

First, set up the Python environment with all dependencies:

```bash
# Submit setup job
./models/hpo/cluster/submit_jobs.sh setup

# Check status
./models/hpo/cluster/submit_jobs.sh status
```

This will:
- Load required modules (Python 3.12.3, PyTorch with CUDA)
- Create a virtual environment
- Install all dependencies from `requirements.txt`
- Verify the installation

### Running HPO Jobs

#### Option 1: Array Job (Recommended)

Run all 9 target configurations in parallel:

```bash
./models/hpo/cluster/submit_jobs.sh array
```

This submits 9 parallel jobs, one for each combination of:
- Flare classes: C, M, X
- Time windows: 24h, 48h, 72h

#### Option 2: Single Target

Run optimization for a specific target:

```bash
# Run M class, 24h window
./models/hpo/cluster/submit_jobs.sh single M 24

# Run X class, 48h window
./models/hpo/cluster/submit_jobs.sh single X 48
```

#### Option 3: CPU-Only Jobs

If GPU resources are limited, use CPU-only jobs (slower but more available):

```bash
./models/hpo/cluster/submit_jobs.sh cpu M 24
```

### Monitoring Jobs

```bash
# Check your job status
./models/hpo/cluster/submit_jobs.sh status

# Or use qstat directly
qstat -u $USER

# Check detailed job info
qstat -f JOB_ID

# View job output (while running)
tail -f job_output_file.out
```

### Managing Jobs

```bash
# Delete a job
qdel JOB_ID

# Delete all your jobs
qstat -u $USER | grep $USER | cut -d. -f1 | xargs qdel
```

## Job Scripts Explained

### 1. `setup_environment.pbs`
- **Resources**: 4 CPU cores, 16GB RAM, 1 hour
- **Purpose**: Sets up Python environment and installs dependencies
- **Queue**: small24

### 2. `run_hpo_single.pbs`
- **Resources**: 8 CPU cores, 64GB RAM, 1 L40S GPU, 24 hours
- **Purpose**: Runs HPO for a single target configuration
- **Queue**: gpu72

### 3. `run_hpo_array.pbs`
- **Resources**: Same as single, but 9 parallel jobs
- **Purpose**: Runs HPO for all 9 target configurations in parallel
- **Queue**: gpu72

### 4. `run_hpo_cpu.pbs`
- **Resources**: 16 CPU cores, 64GB RAM, 48 hours (no GPU)
- **Purpose**: CPU-only HPO for when GPU resources are limited
- **Queue**: medium72

## Resource Requirements and Optimization

### GPU Jobs
- **Recommended**: L40S GPUs (48GB VRAM)
- **Cores**: 8 CPUs per GPU (good CPU:GPU ratio)
- **Memory**: 64GB RAM (sufficient for model and data)
- **Walltime**: 24 hours (should be enough for 166 trials)

### CPU Jobs
- **Cores**: 16 CPUs (for parallel data loading and processing)
- **Memory**: 64GB RAM
- **Walltime**: 48 hours (longer due to slower training)

### Queue Selection
Jobs are automatically routed to appropriate queues based on resource requests:
- GPU jobs → `gpu72` queue
- Single node CPU jobs → `medium24`/`medium72`
- Multi-core CPU jobs → `large24`/`large72`

## Data Management

### Storage Locations

1. **$HOME**: Your home directory (930GB quota)
   - Store code, scripts, and small results
   - Backed up and persistent

2. **$EPHEMERAL**: Temporary storage (10TB quota, 30-day retention)
   - Use for large datasets and intermediate results
   - Faster I/O, but files deleted after 30 days

3. **$TMPDIR**: Job-local storage (200-900GB)
   - Fastest I/O, local to compute node
   - Automatically cleaned up after job ends

### Data Strategy

For large datasets, consider copying to `$TMPDIR` at job start:

```bash
# In your PBS script
echo "Copying data to local storage..."
cp -r $HOME/data/ $TMPDIR/
export DATA_PATH=$TMPDIR/data

# Run your job with local data
python models/hpo/run_hpo.py --data-path $DATA_PATH ...

# Copy results back
cp -r results/ $HOME/
```

## HPO Framework Integration

The cluster scripts work seamlessly with your existing HPO framework:

### Study Configuration
- **3-Stage Protocol**: Exploration (120 trials) → Refinement (40 trials) → Confirmation (6 trials)
- **Total Trials**: 166 per target configuration
- **Timeout**: 23 hours (1 hour buffer for setup/cleanup)

### Output Structure
```
results/
├── hpo_C_24h/          # C class, 24h window results
├── hpo_C_48h/
├── hpo_C_72h/
├── hpo_M_24h/          # M class results
├── hpo_M_48h/
├── hpo_M_72h/
├── hpo_X_24h/          # X class results
├── hpo_X_48h/
└── hpo_X_72h/
```

Each directory contains:
- Optuna study database
- Best hyperparameters
- Optimization history
- Visualization plots

## Troubleshooting

### Common Issues

1. **Job stuck in queue**
   - GPU jobs may take longer to start
   - Consider using CPU jobs or smaller resource requests
   - Check queue status: `qstat -Q`

2. **Out of memory errors**
   - Increase memory request in PBS script
   - Consider using smaller batch sizes
   - Use `$TMPDIR` for data staging

3. **Module loading errors**
   - Ensure you're using `module load tools/prod` first
   - Check available modules: `module avail PyTorch`

4. **Import errors**
   - Verify virtual environment is activated
   - Check that setup job completed successfully
   - Run setup job again if needed

### Debugging

```bash
# Check job logs
ls -la *.out *.err

# View recent output
tail -50 job_name.out
tail -50 job_name.err

# Check GPU usage (if running)
nvidia-smi

# Check disk usage
df -h $HOME
df -h $EPHEMERAL
```

### Getting Help

1. **Imperial RCS Documentation**: Available on their website
2. **RCS Support**: Raise a ticket for technical issues
3. **Job Optimization**: Contact RCS for advice on resource requests

## Best Practices

1. **Test First**: Run a small test job before submitting large arrays
2. **Monitor Resources**: Check CPU/GPU utilization to optimize requests
3. **Use Dependencies**: Chain jobs using PBS dependencies
4. **Backup Results**: Copy important results to permanent storage
5. **Clean Up**: Remove temporary files and old job outputs

## Advanced Usage

### Custom Resource Requests

Modify PBS scripts for different resource needs:

```bash
# For memory-intensive jobs
#PBS -l select=1:ncpus=8:mem=128gb:ngpus=1

# For longer jobs
#PBS -l walltime=72:00:00

# For specific GPU types
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=A100
```

### Job Arrays with Dependencies

```bash
# Submit setup, then array job that depends on it
setup_id=$(qsub models/hpo/cluster/setup_environment.pbs)
qsub -W depend=afterok:$setup_id models/hpo/cluster/run_hpo_array.pbs
```

### Checkpointing Long Jobs

For very long optimizations, consider implementing checkpointing to resume interrupted jobs.

## Performance Expectations

Based on the cluster specifications and your HPO framework:

- **GPU Jobs (L40S)**: ~10-15 minutes per trial → 24 hours for 166 trials
- **CPU Jobs**: ~30-45 minutes per trial → 48+ hours for 166 trials
- **Array Jobs**: All 9 targets complete in parallel (~24 hours total)

This allows you to complete the full HPO study (all 9 targets, 1494 total trials) in about 24 hours using the array job approach. 