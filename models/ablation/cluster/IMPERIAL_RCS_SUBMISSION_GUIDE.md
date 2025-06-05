# Imperial RCS Ablation Study Submission Guide

## 🏛️ **Imperial RCS Specifications**

Based on the current Imperial RCS documentation:

### **GPU Queues Available:**
- `gpu72` - Standard GPU queue
- `gpu72_8` - 8-GPU queue for whole node jobs

### **GPU Types in Batch Queue:**
- **L40S PCIe 48GB** - Ada Lovelace architecture (recommended)
- **A100 PCIe 40GB** - Ampere architecture (limited availability)

### **Maximum Resources:**
- **8 GPUs per node** (not 4 as in old scripts)
- **Up to 32 CPUs per node**
- **Up to 768GB RAM per node**

⚠️ **Important**: RTX6000 GPUs are **NOT available** in batch queues (only in JupyterHub)

## 🚀 **Updated Submission Options**

### **1. Whole Node (8 GPUs) - Maximum Efficiency**

```bash
qsub models/ablation/cluster/submit_whole_node_updated.pbs
```

**Specifications:**
- **Queue**: `gpu72_8`
- **Resources**: `select=1:ncpus=32:mem=128gb:ngpus=8`
- **8 GPUs in parallel** (L40S 48GB or A100 40GB)
- **60 experiments total** (35 component + 25 sequence)
- **~3-6 hours** completion time
- **Walltime**: 24 hours

**When to use:** When you need maximum speed and cluster has whole nodes available

---

### **2. Single GPU - Safe and Reliable**

```bash
qsub models/ablation/cluster/submit_sequential_updated.pbs
```

**Specifications:**
- **Queue**: `gpu72`
- **Resources**: `select=1:ncpus=8:mem=32gb:ngpus=1`
- **1 GPU sequential** (L40S 48GB or A100 40GB)
- **21 experiments** (7 variants × 3 seeds)
- **~24-36 hours** completion time
- **Walltime**: 48 hours

**When to use:** When queue limits are strict or for guaranteed completion

---

### **3. Request Specific GPU Type (Optional)**

If you need a specific GPU type:

```bash
# For L40S (48GB) - recommended
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=L40S

# For A100 (40GB) - limited availability
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=A100
```

**Note**: Leave gpu_type unspecified unless you have a specific requirement.

## 📊 **Performance Comparison**

| Approach | Queue | GPUs | GPU Memory | Experiments | Time | Efficiency |
|----------|-------|------|------------|-------------|------|------------|
| **Whole Node** | gpu72_8 | 8 | 48GB each | 60 | ~4h | Maximum |
| **Single GPU** | gpu72 | 1 | 48GB | 21 | ~30h | Safe |

## 🎯 **Recommended Workflow**

### **Step 1: Check Current Queue Status**
```bash
# SSH to Imperial RCS
ssh your_username@login.hpc.ic.ac.uk

# Check queue status
qstat -Q gpu72
qstat -Q gpu72_8

# Check your current jobs
qstat -u your_username
```

### **Step 2: Choose Submission Strategy**

**If no queue limits and whole nodes available:**
```bash
cd path/to/masters-project
qsub models/ablation/cluster/submit_whole_node_updated.pbs
```

**If queue limits or single GPU preferred:**
```bash
qsub models/ablation/cluster/submit_sequential_updated.pbs
```

### **Step 3: Monitor Progress**
```bash
# Check job status
qstat -u your_username

# Watch logs
tail -f logs/ablation_*.log

# Check GPU usage (when job is running)
ssh compute_node_name
nvidia-smi
```

## 🔧 **Key Differences from Old Scripts**

### **Updated Queue Names:**
- ❌ Old: `#PBS -q v1_gpu72`
- ✅ New: `#PBS -q gpu72` or `#PBS -q gpu72_8`

### **Updated GPU Specifications:**
- ❌ Old: RTX6000 24GB (not available in batch)
- ✅ New: L40S 48GB or A100 40GB

### **Updated Resource Requests:**
- ❌ Old: `ngpus=4` (arbitrary limit)
- ✅ New: `ngpus=8` (actual maximum per node)

### **Dynamic GPU Detection:**
The updated scripts automatically detect the number of allocated GPUs:
```bash
IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
ACTUAL_NUM_GPUS=${#GPU_ARRAY[@]}
```

## 💡 **Pro Tips for Imperial RCS**

1. **Don't specify gpu_type** unless you need a specific GPU
2. **Use gpu72_8 queue** for whole node jobs (8 GPUs)
3. **Use gpu72 queue** for single GPU jobs
4. **L40S GPUs have 48GB memory** - much more than RTX6000 (24GB)
5. **A100 GPUs are limited** - only request if specifically needed
6. **Submit during off-peak hours** for better queue times

## 🆘 **Troubleshooting**

### **Queue Rejection:**
```bash
# Check queue limits
qstat -Q gpu72
qstat -Q gpu72_8

# Check node availability
pbsnodes -a | grep -E "(state|ngpus|gpu_type)"
```

### **Job Fails to Start:**
```bash
# Check job details
qstat -f job_id

# Check if requesting unavailable resources
# (e.g., RTX6000 in batch queue)
```

### **GPU Memory Issues:**
```bash
# L40S has 48GB - should handle batch_size=1024
# If still issues, reduce batch size:
python -m ablation.trainer --variant full_model --seed 0 --batch-size 512
```

## 📈 **Expected Performance**

### **With L40S GPUs (48GB):**
- **Single experiment**: ~45-90 minutes
- **Batch size**: 1024 (full)
- **Memory usage**: ~12-16GB per experiment

### **With A100 GPUs (40GB):**
- **Single experiment**: ~30-60 minutes (faster compute)
- **Batch size**: 1024 (full)
- **Memory usage**: ~12-16GB per experiment

## 🎉 **Expected Timeline**

### **Whole Node Approach (8 GPUs):**
- **Queue time**: 0-4 hours (depending on demand)
- **Execution time**: 3-6 hours
- **Total time**: 3-10 hours

### **Single GPU Approach:**
- **Queue time**: 0-2 hours (usually faster)
- **Execution time**: 24-36 hours
- **Total time**: 24-38 hours

---

**The updated scripts are optimized for the current Imperial RCS infrastructure with L40S/A100 GPUs and correct queue specifications!** 🎯 