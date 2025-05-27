# EVEREST Ablation Study - Array Job Approach

## 🎯 **Strategy: 10 GPUs × 6 Experiments Each**

This approach splits your 60 experiments across **10 separate GPU requests**, where each GPU runs **6 experiments sequentially**. This is much more likely to get resources than requesting 8 GPUs at once.

## 📊 **Experiment Distribution**

- **Total experiments**: 60 (35 component + 25 sequence ablations)
- **Array jobs**: 10 (indexed 1-10)
- **Experiments per GPU**: 6 (some get 7 due to remainder)
- **Resource per job**: 1 GPU, 8 CPUs, 64GB RAM, 12h walltime

## 🚀 **Submission Commands**

### **Option 1: Array Job (Recommended)**
```bash
# Submit the array job (10 separate GPU requests)
qsub models/ablation/cluster/submit_array_correct.pbs
```

### **Option 2: Original Array Job (if routing fails)**
```bash
# Use the original with v1_gpu72 queue
qsub models/ablation/cluster/submit_array_optimized.pbs
```

## 📋 **How It Works**

### **Experiment Assignment**
- **Array job 1**: Experiments 1-7 (7 experiments)
- **Array job 2**: Experiments 8-13 (6 experiments)
- **Array job 3**: Experiments 14-19 (6 experiments)
- **Array job 4**: Experiments 20-25 (6 experiments)
- **Array job 5**: Experiments 26-31 (6 experiments)
- **Array job 6**: Experiments 32-37 (6 experiments)
- **Array job 7**: Experiments 38-43 (6 experiments)
- **Array job 8**: Experiments 44-49 (6 experiments)
- **Array job 9**: Experiments 50-55 (6 experiments)
- **Array job 10**: Experiments 56-60 (5 experiments)

### **Sequential Execution**
Each array job runs its assigned experiments **one at a time**:
1. Load environment and check GPU
2. Run experiment 1 (up to 2 hours)
3. Run experiment 2 (up to 2 hours)
4. Continue until all assigned experiments complete
5. Return GPU to pool

## 🎮 **Monitoring Progress**

### **Check Job Status**
```bash
# Check all array jobs
qstat -u $USER

# Check specific array job
qstat -t 1168733  # Replace with your job ID
```

### **Monitor Logs**
```bash
# Watch logs from all array jobs
tail -f logs/ablation_array_*_*.log

# Watch specific array job
tail -f logs/ablation_array_1168733_1.log  # Array index 1
```

### **Check Results**
```bash
# Count completed experiments
ls models/ablation/results/ | wc -l

# Check specific results
ls -la models/ablation/results/ablation_*
```

## ⏱️ **Expected Timeline**

### **Best Case Scenario**
- **All 10 jobs start immediately**: ~6-8 hours total
- **Each experiment**: ~45-60 minutes
- **Parallel execution**: Maximum efficiency

### **Realistic Scenario**
- **Jobs start gradually**: ~8-12 hours total
- **Some queue time**: Jobs start as resources become available
- **High success rate**: Single GPU requests are prioritized

## 🔧 **Advantages of Array Job Approach**

### ✅ **Benefits**
- **Higher success rate**: Single GPU requests are easier to fulfill
- **Fault tolerance**: If one job fails, others continue
- **Efficient resource use**: No wasted GPU time
- **Queue-friendly**: Doesn't monopolize resources
- **Scalable**: Can adjust number of array jobs easily

### ⚠️ **Considerations**
- **More complex monitoring**: 10 separate jobs to track
- **Potential queue delays**: Some jobs may start later
- **Log management**: Multiple log files to check

## 🚨 **Troubleshooting**

### **If Array Jobs Are Rejected**
```bash
# Check queue limits for array jobs
qstat -Qf v1_gpu72 | grep -i array

# Try smaller array size
# Edit script: #PBS -J 1-5 (5 jobs × 12 experiments each)
```

### **If Some Jobs Fail**
```bash
# Identify failed experiments
grep -l "❌" logs/ablation_array_*_*.log

# Resubmit specific array indices
qsub -J 3,7,9 models/ablation/cluster/submit_array_correct.pbs
```

### **If Queue Time Is Too Long**
```bash
# Fall back to single GPU sequential
qsub models/ablation/cluster/submit_single_gpu_correct.pbs
```

## 📈 **Performance Comparison**

| Approach | GPUs | Jobs | Time | Queue Risk | Efficiency |
|----------|------|------|------|------------|------------|
| Whole Node | 8 | 1 | ~8h | High | Maximum |
| Array Job | 10 | 10 | ~8h | Medium | High |
| Single GPU | 1 | 1 | ~36h | Low | Good |

## 🎯 **Recommendation**

**Try the array job approach first** - it offers the best balance of efficiency and success rate. If it gets rejected or queues too long, fall back to the single GPU approach.

## 📞 **Next Steps After Submission**

1. **Monitor job status**: `qstat -u $USER`
2. **Watch logs**: `tail -f logs/ablation_array_*_*.log`
3. **Check progress**: `ls models/ablation/results/ | wc -l`
4. **Run analysis**: After all jobs complete, run `python run_updated_ablation.py --analysis-only`

---

**This array job approach maximizes your chances of getting all 60 experiments completed efficiently!** 