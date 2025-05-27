# Efficient Ablation Study Submission Strategies

## 🎯 **The Problem with Array Jobs**

You're absolutely right! The original array job approach is inefficient:

```bash
#PBS -J 1-60  # Requests 60 separate GPUs!
```

**Problems:**
- ❌ Each job requests a separate GPU
- ❌ GPU setup overhead for each experiment  
- ❌ Queue limits prevent submission
- ❌ Inefficient resource utilization
- ❌ Long queue wait times

## ✅ **Better Approaches**

### **1. Sequential Execution (Recommended for Queue Limits)**

**Script:** `submit_sequential_batch.pbs`

```bash
# Single GPU, run experiments sequentially
qsub models/ablation/cluster/submit_sequential_batch.pbs
```

**Advantages:**
- ✅ Only requests 1 GPU
- ✅ Runs 21 experiments sequentially (7 variants × 3 seeds)
- ✅ Works within queue limits
- ✅ 48-hour walltime for completion
- ✅ Error handling and progress tracking

**Resource Request:**
```bash
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -l walltime=48:00:00
```

**Expected Runtime:** ~24-36 hours for 21 experiments

---

### **2. Whole Node Approach (Best for Efficiency)**

**Script:** `submit_whole_node.pbs`

```bash
# Request full node with 4 GPUs
qsub models/ablation/cluster/submit_whole_node.pbs
```

**Advantages:**
- ✅ Maximum efficiency: 4 GPUs in parallel
- ✅ Completes all 60 experiments in ~6-12 hours
- ✅ Only 1 job submission (no queue limits)
- ✅ Optimal resource utilization

**Resource Request:**
```bash
#PBS -l select=1:ncpus=32:mem=128gb:ngpus=4
#PBS -l walltime=24:00:00
```

**Expected Runtime:** ~6-12 hours for 60 experiments

---

### **3. Small Batch Arrays (Compromise)**

**Script:** `submit_ablation_small.pbs`

```bash
# 5 separate GPUs for core experiments
qsub models/ablation/cluster/submit_ablation_small.pbs
```

**Advantages:**
- ✅ Works within most queue limits
- ✅ Some parallelization (5 jobs)
- ✅ Quick results for core ablations

**Resource Request:**
```bash
#PBS -J 1-5  # Only 5 jobs
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1
```

## 📊 **Comparison Table**

| Approach | GPUs | Jobs | Runtime | Queue Risk | Efficiency |
|----------|------|------|---------|------------|------------|
| **Array (Original)** | 60 | 60 | ~2-4h | ❌ High | ❌ Low |
| **Sequential** | 1 | 1 | ~36h | ✅ None | ⚠️ Medium |
| **Whole Node** | 4 | 1 | ~8h | ⚠️ Medium | ✅ High |
| **Small Batch** | 5 | 5 | ~2h | ✅ Low | ⚠️ Medium |

## 🚀 **Recommended Strategy**

### **For ICL Cluster (Your Situation):**

**Option A: Start with Sequential (Safe)**
```bash
qsub models/ablation/cluster/submit_sequential_batch.pbs
```

**Option B: Try Whole Node (Optimal)**
```bash
qsub models/ablation/cluster/submit_whole_node.pbs
```

### **Decision Tree:**

```
Can you get a whole node? 
├─ YES → Use submit_whole_node.pbs (4 GPUs, ~8 hours)
└─ NO → Use submit_sequential_batch.pbs (1 GPU, ~36 hours)
```

## 🔧 **Implementation Details**

### **Sequential Approach Features:**
- **Timeout protection:** 2 hours per experiment
- **Error handling:** Continues if one experiment fails
- **Progress tracking:** Shows completion status
- **Walltime monitoring:** Stops early if time running out
- **GPU cooling:** 30-second pause between experiments

### **Whole Node Features:**
- **Parallel execution:** 4 experiments simultaneously
- **Load balancing:** Distributes experiments across GPUs
- **Process management:** Tracks all GPU processes
- **Resource monitoring:** Full node utilization

## 💡 **Why This is Better**

### **Resource Efficiency:**
```
Array Jobs:    60 × (GPU setup + experiment + teardown)
Sequential:    1 × (GPU setup) + 21 × (experiment) + 1 × (teardown)
Whole Node:    1 × (4 GPU setup) + 15 × (4 parallel experiments) + 1 × (teardown)
```

### **Queue Efficiency:**
```
Array Jobs:    60 queue slots
Sequential:    1 queue slot
Whole Node:    1 queue slot (but larger resource request)
```

### **Time Efficiency:**
```
Array Jobs:    Limited by queue wait time + slowest job
Sequential:    Sum of all experiment times
Whole Node:    Sum of experiment times ÷ 4
```

## 🎯 **Quick Start Commands**

### **Test First (Single Experiment):**
```bash
cd models/ablation
python -m ablation.trainer --variant full_model --seed 0
```

### **Submit Sequential Batch:**
```bash
qsub models/ablation/cluster/submit_sequential_batch.pbs
```

### **Submit Whole Node (if available):**
```bash
qsub models/ablation/cluster/submit_whole_node.pbs
```

### **Monitor Progress:**
```bash
# Check job status
qstat -u $USER

# Watch logs
tail -f logs/ablation_*.log

# Check results
ls models/ablation/results/
```

## 🔍 **Monitoring and Debugging**

### **Sequential Job Monitoring:**
```bash
# Watch progress
tail -f logs/ablation_sequential_*.log

# Check GPU usage
ssh <node> nvidia-smi
```

### **Whole Node Monitoring:**
```bash
# Watch all GPU logs
tail -f logs/ablation_node_*.log

# Check all GPU usage
ssh <node> watch -n 1 nvidia-smi
```

## 📈 **Expected Results**

### **Sequential Approach:**
- **21 experiments** (7 variants × 3 seeds)
- **~36 hours** total runtime
- **1 GPU** utilized
- **Guaranteed completion** within queue limits

### **Whole Node Approach:**
- **60 experiments** (7 variants × 5 seeds + 5 sequence × 5 seeds)
- **~8 hours** total runtime  
- **4 GPUs** utilized
- **Maximum efficiency** if node available

---

**The whole node approach is definitely the best if you can get it!** 🎯

It's much more efficient than array jobs and maximizes GPU utilization while minimizing queue overhead. 