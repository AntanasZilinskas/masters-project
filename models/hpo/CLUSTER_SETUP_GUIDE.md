# Imperial RCS Cluster Setup Guide

This guide provides step-by-step instructions for setting up and running your EVEREST solar flare prediction HPO study on the Imperial College London Research Computing Service (RCS) cluster.

## Table of Contents

1. [Initial Setup](#initial-setup)
2. [Data Preparation](#data-preparation)
3. [Environment Setup](#environment-setup)
4. [Running HPO Jobs](#running-hpo-jobs)
5. [Monitoring and Management](#monitoring-and-management)
6. [Results Collection](#results-collection)
7. [Troubleshooting](#troubleshooting)

## Initial Setup

### 1. Access the Cluster

Connect to the Imperial RCS login node:

```bash
ssh your_username@login.cx3.hpc.imperial.ac.uk
```

### 2. Upload Your Project

From your local machine, upload the entire project directory:

```bash
# Upload project files
scp -r /path/to/masters-project your_username@login.cx3.hpc.imperial.ac.uk:~/

# Or using rsync for large projects
rsync -avz --progress /path/to/masters-project/ your_username@login.cx3.hpc.imperial.ac.uk:~/masters-project/
```

### 3. Initial Cluster Setup

Once logged into the cluster:

```bash
# Navigate to your project ROOT directory (CRITICAL)
cd ~/masters-project

# Verify you're in the correct location
ls -la  # Should show: data/ models/ requirements.txt etc.
pwd     # Should be: /rds/general/user/USERNAME/home/masters-project

# Make scripts executable
chmod +x models/hpo/cluster/submit_jobs.sh models/hpo/cluster/monitor_jobs.sh

# Check cluster resources
qstat -Q  # View queue status
module avail | grep Python  # Check available Python modules
```

**Important**: Always run cluster scripts from the project root directory (`~/masters-project/`), NOT from inside the `models/hpo/cluster/` directory. This prevents module import errors.

## Data Preparation

### Data Location Strategy

Choose the appropriate storage location based on your data size:

#### Small datasets (< 10GB)
Store in your home directory (`$HOME`):
```bash
mkdir -p ~/masters-project/data
# Copy data here
```

#### Large datasets (10GB - 1TB)
Use ephemeral storage (`$EPHEMERAL`):
```bash
mkdir -p $EPHEMERAL/solar_flare_data
# Copy large datasets here
# Update your data paths in configuration files
```

#### Example Data Structure
```
data/
├── training/
│   ├── C_class_24h.npz
│   ├── M_class_24h.npz
│   └── X_class_24h.npz
├── validation/
└── test/
```

### Data Transfer Tips

For large datasets:
```bash
# Use compression for transfer
tar -czf solar_data.tar.gz data/
scp solar_data.tar.gz username@login.cx3.hpc.imperial.ac.uk:$EPHEMERAL/

# On cluster: extract
cd $EPHEMERAL
tar -xzf solar_data.tar.gz
```

## Environment Setup

### 1. Automated Setup (Recommended)

Run the complete setup workflow:

```bash
./models/hpo/cluster/submit_jobs.sh setup
```

This will:
- Load Python 3.12.3 and PyTorch modules
- Create a virtual environment (`venv_hpo`)
- Install all dependencies from `requirements.txt`
- Verify the installation

### 2. Monitor Setup Progress

```bash
# Check job status
./models/hpo/cluster/monitor_jobs.sh status

# View setup logs
./models/hpo/cluster/monitor_jobs.sh logs setup
```

### 3. Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Load modules
module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Create virtual environment
python -m venv venv_hpo
source venv_hpo/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Test installation
python -c "import optuna, torch; print('Setup successful!')"
```

## Running HPO Jobs

### Option 1: Complete Workflow (Recommended)

Run all 9 target configurations in parallel:

```bash
./models/hpo/cluster/submit_jobs.sh all
```

This submits:
- 1 setup job (if needed)
- 9 parallel HPO jobs (array job)

### Option 2: Array Job Only

If environment is already set up:

```bash
./models/hpo/cluster/submit_jobs.sh array
```

### Option 3: Individual Targets

Run specific target configurations:

```bash
# M class, 24h window
./models/hpo/cluster/submit_jobs.sh single M 24

# X class, 48h window  
./models/hpo/cluster/submit_jobs.sh single X 48

# C class, 72h window
./models/hpo/cluster/submit_jobs.sh single C 72
```

### Option 4: CPU-Only Jobs

If GPU resources are limited:

```bash
./models/hpo/cluster/submit_jobs.sh cpu M 24
```

### Resource Requirements Summary

| Job Type | CPUs | Memory | GPU | Walltime | Queue |
|----------|------|--------|-----|----------|-------|
| Setup | 4 | 16GB | - | 1h | small24 |
| GPU HPO | 8 | 64GB | 1 L40S | 24h | gpu72 |
| CPU HPO | 16 | 64GB | - | 48h | medium72 |

## Monitoring and Management

### Real-time Monitoring

```bash
# Watch job status in real-time
./models/hpo/cluster/monitor_jobs.sh watch

# Check current status
./models/hpo/cluster/monitor_jobs.sh status

# View recent logs
./models/hpo/cluster/monitor_jobs.sh logs

# Check resource usage
./models/hpo/cluster/monitor_jobs.sh resources
```

### Job Management Commands

```bash
# View all your jobs
qstat -u $USER

# Get detailed job info
qstat -f JOB_ID

# Cancel a job
qdel JOB_ID

# Cancel all your jobs
qstat -u $USER | grep $USER | cut -d. -f1 | xargs qdel
```

### Expected Timeline

- **Setup job**: 5-15 minutes
- **Single HPO job**: 18-24 hours (166 trials)
- **Array job**: 18-24 hours (all 9 targets in parallel)
- **CPU job**: 36-48 hours (slower training)

## Results Collection

### Result Structure

After completion, your results will be organized as:

```
results/
├── hpo_C_24h/
│   ├── study.db           # Optuna study database
│   ├── best_params.json   # Best hyperparameters
│   ├── optimization_history.png
│   └── parameter_importance.png
├── hpo_C_48h/
├── hpo_C_72h/
├── hpo_M_24h/
├── hpo_M_48h/
├── hpo_M_72h/
├── hpo_X_24h/
├── hpo_X_48h/
└── hpo_X_72h/
```

### Downloading Results

```bash
# From your local machine
scp -r username@login.cx3.hpc.imperial.ac.uk:~/masters-project/results/ ./

# Or using rsync
rsync -avz username@login.cx3.hpc.imperial.ac.uk:~/masters-project/results/ ./results/
```

### Analyzing Results

Once downloaded locally:

```bash
# Generate summary report
python models/hpo/run_hpo.py --target multi --analyze-results results/

# Create combined visualizations
python -m models.hpo.visualization --input-dir results/ --output-dir analysis/
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Job Stuck in Queue
```bash
# Check queue status
qstat -Q

# Try smaller resource request or CPU-only job
./models/hpo/cluster/submit_jobs.sh cpu M 24
```

#### 2. Out of Memory Errors
```bash
# Increase memory in PBS script
#PBS -l select=1:ncpus=8:mem=128gb:ngpus=1

# Or reduce batch size in your model configuration
```

#### 3. Module Loading Errors
```bash
# Check available modules
module avail Python
module avail PyTorch

# Load tools/prod first
module load tools/prod
```

#### 4. Import Errors
```bash
# Verify virtual environment
source venv_hpo/bin/activate
python -c "from models.hpo import HPOObjective"

# Re-run setup if needed
./models/hpo/cluster/submit_jobs.sh setup
```

#### 5. Data Access Issues
```bash
# Check data location
ls -la data/
ls -la $EPHEMERAL/

# Verify paths in configuration
python -c "import os; print(os.path.exists('data/training/'))"
```

### Performance Optimization

#### GPU Utilization
```bash
# Check GPU usage during job
ssh to_compute_node  # Use job output to find node
nvidia-smi
```

#### Memory Usage
```bash
# Monitor memory in running job
qstat -f JOB_ID | grep resources_used
```

#### I/O Optimization
For large datasets, modify PBS scripts to use `$TMPDIR`:

```bash
# Add to PBS script after cd $PBS_O_WORKDIR
cp -r data/ $TMPDIR/
export DATA_PATH=$TMPDIR/data

# Run with local data
python models/hpo/run_hpo.py --data-path $DATA_PATH ...

# Copy results back
cp -r results/ $PBS_O_WORKDIR/
```

### Getting Help

1. **Check Documentation**: Review `models/hpo/cluster/README.md`
2. **Imperial RCS Support**: Raise a ticket for cluster issues
3. **Job Optimization**: Contact RCS for resource request advice
4. **Framework Issues**: Review HPO framework logs and documentation

### Maintenance Commands

```bash
# Clean up old files
./models/hpo/cluster/monitor_jobs.sh cleanup

# Check disk usage
df -h $HOME
df -h $EPHEMERAL

# Archive completed results
tar -czf hpo_results_$(date +%Y%m%d).tar.gz results/
```

## Next Steps

After successful HPO completion:

1. **Analyze Results**: Compare performance across targets
2. **Select Best Models**: Extract optimal hyperparameters
3. **Final Training**: Train final models with best parameters
4. **Evaluation**: Validate on test sets
5. **Documentation**: Update your thesis with cluster results

## Resource Limits and Fair Use

### Cluster Limits
- **GPU Jobs**: 12 GPUs total per user
- **Walltime**: 72 hours maximum
- **Storage**: 930GB home, 10TB ephemeral

### Best Practices
- Submit jobs during off-peak hours when possible
- Use appropriate resource requests (don't over-allocate)
- Clean up temporary files regularly
- Monitor resource usage and optimize accordingly

This completes your cluster setup guide. The HPO framework is now ready to run at scale on the Imperial RCS cluster! 