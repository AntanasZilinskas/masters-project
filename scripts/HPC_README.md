# EVEREST Model Training on Imperial HPC

This document outlines the complete process for training the EVEREST model on Imperial College London's Research Computing Service (RCS).

## Prerequisites

Before you begin:

1. Join Imperial's RCS/HPC service by completing their web form
2. Install Imperial's "Unified Access" VPN for off-site access
3. Ensure your code repository is accessible (GitHub, GitLab, etc.)

## Step 1: Set Up Your Environment

### Local Machine Setup

Before connecting to the HPC, prepare your data transfer script:

1. Edit `scripts/transfer_data.sh`:
   - Set `IMPERIAL_USER` to your Imperial College username
   - Set `RDS_PROJECT_CODE` to your assigned RDS project code
   - Set `LOCAL_DATA_PATH` to the path of your data directory

2. Make the script executable:
   ```bash
   chmod +x scripts/transfer_data.sh
   ```

3. Connect to Imperial's VPN and run the transfer script:
   ```bash
   ./scripts/transfer_data.sh
   ```

### HPC Environment Setup

1. SSH into the HPC:
   ```bash
   ssh your_username@login.hpc.imperial.ac.uk
   ```

2. Set up your project workspace:
   ```bash
   mkdir -p $HOME/projects/everest/{src,data,results,logs,scripts}
   cd $HOME/projects/everest
   ```

3. Clone your repository:
   ```bash
   git clone https://github.com/your-username/masters-project.git src
   ```

4. Set up your Python environment:
   ```bash
   module load miniforge/3
   miniforge-setup
   eval "$(~/miniforge3/bin/conda shell.bash hook)"
   conda create -n everest_env -c conda-forge python=3.11 cudatoolkit=11.8
   conda activate everest_env
   pip install -r $HOME/projects/everest/src/requirements_full.txt
   conda config --set auto_activate_base false
   ```

5. Copy the PBS job script:
   ```bash
   cp $HOME/projects/everest/src/scripts/everest_train.pbs $HOME/projects/everest/scripts/
   ```

## Step 2: Customize the Training Job

Edit the PBS script to match your specific training requirements:

```bash
cd $HOME/projects/everest/scripts
nano everest_train.pbs
```

Key parameters to consider changing:
- `#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=L40S` - Resource allocation
- `#PBS -l walltime=12:00:00` - Maximum job runtime
- Python script parameters (`--flare`, `--window`, `--epochs`, etc.)

## Step 3: Submit and Monitor the Job

```bash
cd $HOME/projects/everest/scripts
qsub everest_train.pbs             # Submit the job
qstat -u $USER                     # Check job status
qstat -f <jobid>                   # Get detailed job info
```

## Step 4: View Results

After the job completes:

```bash
cd $HOME/projects/everest/results/<jobid>
```

To copy results back to your local machine:

```bash
# Run this on your local machine
scp -r your_username@login.hpc.imperial.ac.uk:$HOME/projects/everest/results/<jobid> ./local_results
```

## Advanced Options

### Multi-GPU Training

To use multiple GPUs, modify the PBS script:

```bash
#PBS -l select=1:ncpus=16:mem=64gb:ngpus=2:gpu_type=L40S
```

And update the Python command to use distributed training:

```bash
python $HOME/projects/everest/src/models/train_complete_everest.py \
       --flare M5 \
       --window 24 \
       --distributed
```

### Hyperparameter Sweeps

For hyperparameter sweeps, use job arrays:

```bash
#PBS -J 0-9
```

And modify the Python command to use the array index:

```bash
python $HOME/projects/everest/src/models/train_complete_everest.py \
       --flare M5 \
       --window 24 \
       --param_sweep ${PBS_ARRAY_INDEX}
``` 