# Optimized Ablation Study Submission Guide

## 🔍 **Check Node Availability First**

Before submitting, check what resources are available:

```bash
# Run the availability checker
chmod +x models/ablation/cluster/check_node_availability.sh
./models/ablation/cluster/check_node_availability.sh
```

This will show you:
- Current queue status
- Your existing jobs
- Available nodes and GPUs
- Recommended submission strategy

## 🚀 **Optimized Submission Options**

### **1. Whole Node (Best Efficiency) - 4 GPUs**

```bash
qsub models/ablation/cluster/submit_whole_node.pbs
```

**Specs:**
- **4 GPUs in parallel**
- **60 experiments total** (35 component + 25 sequence)
- **~6-12 hours** completion time
- **32 CPUs, 128GB RAM**

**When to use:** When cluster has whole nodes available

---

### **2. Optimized Sequential (Memory Sharing) - 1 GPU**

```bash
qsub models/ablation/cluster/submit_sequential_optimized.pbs
```

**Specs:**
- **1 GPU with 3 parallel processes**
- **35 experiments** (7 variants × 5 seeds)
- **~12-24 hours** completion time
- **16 CPUs, 64GB RAM**
- **Reduced batch size (512) for memory sharing**

**When to use:** When you want faster than sequential but can't get whole node

---

### **3. Standard Sequential (Safe) - 1 GPU**

```bash
qsub models/ablation/cluster/submit_sequential_batch.pbs
```

**Specs:**
- **1 GPU, 1 process at a time**
- **21 experiments** (7 variants × 3 seeds)
- **~24-36 hours** completion time
- **8 CPUs, 32GB RAM**

**When to use:** When queue limits are strict or for guaranteed completion

---

### **4. Small Batch (Quick Test) - 5 GPUs**

```bash
qsub models/ablation/cluster/submit_ablation_small.pbs
```

**Specs:**
- **5 separate GPUs**
- **5 core experiments** (1 seed each)
- **~2-4 hours** completion time

**When to use:** For quick testing or when queue limits are very strict

## 📊 **Performance Comparison**

| Approach | GPUs | Parallel | Experiments | Time | Memory | Queue Risk |
|----------|------|----------|-------------|------|--------|------------|
| **Whole Node** | 4 | 4 | 60 | ~8h | High | Medium |
| **Optimized Sequential** | 1 | 3 | 35 | ~18h | Medium | Low |
| **Standard Sequential** | 1 | 1 | 21 | ~30h | Low | None |
| **Small Batch** | 5 | 5 | 5 | ~3h | Low | Low |

## 🧠 **Memory Optimization Features**

### **Optimized Sequential Approach:**
- **3 experiments in parallel** on single GPU
- **Reduced batch size (512)** instead of 1024
- **GPU memory sharing** between processes
- **Process isolation** with separate CUDA contexts

### **How it works:**
```bash
# Each process gets reduced batch size
python -m ablation.trainer --variant full_model --seed 0 --batch-size 512 --memory-efficient

# 3 processes run simultaneously:
# Process 0: full_model, seed 0
# Process 1: no_evidential, seed 1  
# Process 2: no_evt, seed 2
```

## 🎯 **Decision Tree**

```
Check node availability first:
├─ Whole node available?
│  ├─ YES → submit_whole_node.pbs (4 GPUs, ~8h)
│  └─ NO → Continue below
├─ Need fast results?
│  ├─ YES → submit_sequential_optimized.pbs (1 GPU, 3 parallel, ~18h)
│  └─ NO → Continue below
├─ Queue limits strict?
│  ├─ YES → submit_sequential_batch.pbs (1 GPU, 1 process, ~30h)
│  └─ NO → submit_sequential_optimized.pbs
└─ Just testing?
   └─ YES → submit_ablation_small.pbs (5 GPUs, ~3h)
```

## 🔧 **Manual Optimization**

You can also run experiments manually with custom settings:

### **Single Experiment with Custom Batch Size:**
```bash
cd models/ablation
python -m ablation.trainer --variant full_model --seed 0 --batch-size 256
```

### **Memory Efficient Mode:**
```bash
python -m ablation.trainer --variant no_evidential --seed 1 --memory-efficient
```

### **Multiple Experiments in Background:**
```bash
# Start 3 experiments in parallel (manually)
python -m ablation.trainer --variant full_model --seed 0 --memory-efficient &
python -m ablation.trainer --variant no_evidential --seed 0 --memory-efficient &
python -m ablation.trainer --variant no_evt --seed 0 --memory-efficient &
wait  # Wait for all to complete
```

## 📈 **Expected GPU Memory Usage**

| Configuration | Batch Size | GPU Memory | Parallel Processes |
|---------------|------------|------------|-------------------|
| **Standard** | 1024 | ~8-12GB | 1 |
| **Optimized** | 512 | ~4-6GB | 3 |
| **Memory Efficient** | 256 | ~2-3GB | Up to 6 |

## 🔍 **Monitoring Commands**

### **Check Job Status:**
```bash
# Your jobs
qstat -u $USER

# Detailed job info
qstat -f <job_id>

# Watch continuously
watch -n 30 'qstat -u $USER'
```

### **Monitor GPU Usage:**
```bash
# SSH to compute node (when job is running)
ssh <node_name>
watch -n 5 nvidia-smi

# Check memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### **Check Logs:**
```bash
# Sequential job
tail -f logs/ablation_sequential_*.log

# Optimized sequential
tail -f logs/ablation_sequential_opt_*.log

# Whole node
tail -f logs/ablation_node_*.log
```

## 🎯 **Recommended Workflow**

### **Step 1: Check Availability**
```bash
./models/ablation/cluster/check_node_availability.sh
```

### **Step 2: Choose Strategy**
Based on the output, pick the best approach:

**If whole node available:**
```bash
qsub models/ablation/cluster/submit_whole_node.pbs
```

**If single GPU only:**
```bash
qsub models/ablation/cluster/submit_sequential_optimized.pbs
```

**If queue limits strict:**
```bash
qsub models/ablation/cluster/submit_sequential_batch.pbs
```

### **Step 3: Monitor Progress**
```bash
# Check job status
qstat -u $USER

# Watch logs
tail -f logs/ablation_*.log

# Check results
ls models/ablation/results/
```

### **Step 4: Run Analysis**
```bash
# After experiments complete
cd models/ablation
python run_updated_ablation.py --analysis-only
```

## 💡 **Pro Tips**

1. **Start with availability check** - saves time and frustration
2. **Use optimized sequential** - good balance of speed and resource usage
3. **Monitor GPU memory** - ensure you're not hitting limits
4. **Submit during off-peak hours** - better chance of getting whole node
5. **Test with small batch first** - verify everything works before full run

## 🆘 **Troubleshooting**

### **GPU Memory Issues:**
```bash
# Reduce batch size further
python -m ablation.trainer --variant full_model --seed 0 --batch-size 128
```

### **Queue Limits:**
```bash
# Use smallest possible submission
qsub models/ablation/cluster/submit_ablation_small.pbs
```

### **Slow Training:**
```bash
# Check if using GPU
nvidia-smi

# Check CPU usage
htop
```

---

**The optimized approaches give you much better GPU utilization while working within cluster constraints!** 🎯 