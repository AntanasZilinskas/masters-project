# Imperial RCS Ablation Study - Correct Submission Guide

## 🏛️ **Imperial RCS Key Facts**

Based on the official Imperial RCS documentation:

### **No Queue Names Required**
- Imperial RCS uses **routing queues**
- You specify resources, PBS automatically places job in appropriate queue
- **Never specify `-q queue_name`** - this causes "Unknown queue name" errors

### **Available GPU Types in Batch Queue:**
- **L40S PCIe 48GB** - Ada Lovelace (recommended)
- **A100 PCIe 40GB** - Ampere (limited availability)
- **A40 and RTX6000** - Only available in JupyterHub, NOT batch queue

### **Resource Limits:**
- **Maximum 8 GPUs per node**
- **Maximum 64 CPUs per node** (some nodes have 128)
- **Maximum 450GB RAM per node** (some nodes have 920GB)
- **GPU job limit: 12 GPUs total per user**

## 🚀 **Corrected Submission Scripts**

### **Option 1: Whole Node (8 GPUs) - Maximum Efficiency**

```bash
qsub models/ablation/cluster/submit_whole_node_correct.pbs
```

**Resource Request:**
- 8 GPUs, 64 CPUs, 450GB RAM
- 24-hour walltime
- Auto-routed to `large24` queue
- ~60 experiments in 6-12 hours

### **Option 2: Single GPU - Queue-Safe**

```bash
qsub models/ablation/cluster/submit_single_gpu_correct.pbs
```

**Resource Request:**
- 1 GPU, 16 CPUs, 128GB RAM
- 48-hour walltime
- Auto-routed to `medium72` queue
- ~21 experiments in 24-36 hours

## 📋 **Queue Routing System**

Imperial RCS automatically routes jobs based on resources:

| Resources | Auto-Routes To | Use Case |
|-----------|----------------|----------|
| 1-16 CPUs, ≤128GB, ≤24h | `small24` | Small jobs |
| 1-16 CPUs, ≤128GB, 24-72h | `small72` | Small long jobs |
| 1-64 CPUs, ≤450GB, ≤24h | `medium24` | Single node |
| 1-64 CPUs, ≤450GB, 24-72h | `medium72` | Single node long |
| 1-128 CPUs, ≤920GB, ≤24h | `large24` | Whole node |
| 1-128 CPUs, ≤920GB, 24-72h | `large72` | Whole node long |
| Any + GPUs, ≤72h | `gpu72` | GPU jobs |

## 🎯 **Recommended Approach**

### **Step 1: Check Availability**
```bash
# On the cluster
bash models/ablation/cluster/check_available_queues.sh
```

### **Step 2: Submit Whole Node First**
```bash
# Try the most efficient approach first
qsub models/ablation/cluster/submit_whole_node_correct.pbs
```

### **Step 3: Fallback to Single GPU if Needed**
```bash
# If whole node is rejected or queued too long
qsub models/ablation/cluster/submit_single_gpu_correct.pbs
```

## 🔧 **Key Differences from Previous Scripts**

### ❌ **What Was Wrong:**
```bash
#PBS -q v1_gpu72        # Queue names don't exist
#PBS -q gpu72_8         # Queue names don't exist  
#PBS -q gpu72           # Queue names don't exist
```

### ✅ **What's Correct:**
```bash
#PBS -l select=1:ncpus=64:mem=450gb:ngpus=8
# No -q flag - PBS routes automatically
```

## 📊 **Resource Optimization**

### **Whole Node Approach:**
- **Efficiency**: 8 GPUs in parallel
- **Time**: ~8 hours for 60 experiments
- **Risk**: May queue longer due to resource requirements

### **Single GPU Approach:**
- **Efficiency**: Sequential execution
- **Time**: ~36 hours for 21 experiments
- **Risk**: Very low, almost always starts quickly

## 🧪 **Testing Commands**

```bash
# Test resource availability (dry run)
qsub -n models/ablation/cluster/submit_whole_node_correct.pbs

# Check job status
qstat -u $USER

# Monitor job progress
tail -f logs/ablation_*_${JOB_ID}.log
```

## 🚨 **Common Issues & Solutions**

### **Issue: "Unknown queue name"**
- **Cause**: Using `-q queue_name` 
- **Solution**: Remove all `-q` flags, let PBS route automatically

### **Issue: Job rejected for resources**
- **Cause**: Requesting too many resources
- **Solution**: Use single GPU script instead

### **Issue: Long queue time**
- **Cause**: High resource request during busy periods
- **Solution**: Submit during off-peak hours or use smaller resources

## 📈 **Performance Comparison**

| Approach | GPUs | Time | Queue Risk | Efficiency |
|----------|------|------|------------|------------|
| Whole Node | 8 | ~8h | Medium | Maximum |
| Single GPU | 1 | ~36h | Minimal | Good |

## 🎯 **Final Recommendations**

1. **Always try whole node first** - maximum efficiency
2. **No queue names** - let PBS routing handle it
3. **Monitor queue times** - switch to single GPU if needed
4. **Use updated hyperparameters** - embed_dim=64, num_blocks=8
5. **Check GPU types** - expect L40S (48GB) or A100 (40GB)

## 📞 **Support**

If jobs are still rejected:
1. Check Imperial RCS documentation updates
2. Contact Imperial RCS support
3. Use the queue checking script to debug

---

**This guide is based on the official Imperial RCS documentation and should work correctly with the current system.** 