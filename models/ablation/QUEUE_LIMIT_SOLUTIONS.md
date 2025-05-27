# ICL Cluster Queue Limit Solutions

## 🚨 Problem
```
qsub: would exceed queue generic's per-user limit
```

The ICL cluster has per-user job limits that prevent submitting all 60 ablation experiments at once.

## ✅ Solutions (Choose One)

### 1. **Small Batch Approach (Recommended)**

Submit just 5 jobs at a time:

```bash
# Submit first small batch (5 jobs)
qsub models/ablation/cluster/submit_ablation_small.pbs

# Check status
qstat -u $USER

# When complete, submit next batch manually
# (Edit the script to change variants or seeds)
```

### 2. **Use Existing Small Batch Script**

```bash
# Submit 10 jobs at a time
qsub models/ablation/cluster/submit_ablation_small_batch.pbs
```

### 3. **Batched Submission Script**

```bash
cd models/ablation/cluster

# Check what would be submitted
./submit_batched.sh --dry-run

# Submit first batch of 10
./submit_batched.sh --batch-size 10

# Submit next batch when first completes
./submit_batched.sh --start-batch 2 --batch-size 10
```

### 4. **Check Queue Limits First**

```bash
# Check current queue status
qstat -u $USER

# Check queue limits
qstat -Q

# See how many jobs you can submit
qstat -u $USER | wc -l
```

### 5. **Single Job Testing**

Test one experiment first:

```bash
# Run single experiment locally
cd models/ablation
python -m ablation.trainer --variant full_model --seed 0

# Or submit single job
qsub -v VARIANT=full_model,SEED=0 models/ablation/cluster/submit_single.pbs
```

## 🎯 **Recommended Workflow for ICL**

### Step 1: Start Small
```bash
# Submit 5 most important variants first
qsub models/ablation/cluster/submit_ablation_small.pbs
```

This runs:
- `full_model` (baseline)
- `no_evidential` (remove NIG head)
- `no_evt` (remove EVT head)  
- `mean_pool` (remove attention)
- `cross_entropy` (remove focal loss)

### Step 2: Monitor Progress
```bash
# Check job status
qstat -u $USER

# Check logs
tail -f logs/ablation_*.log

# Check results
ls models/ablation/results/
```

### Step 3: Submit Additional Batches
```bash
# When first batch completes, submit remaining variants
# Edit submit_ablation_small.pbs to change variants:
# VARIANTS=("no_precursor" "fp32_training" "full_model" "no_evidential" "no_evt")
# SEED=1  # Use different seed

qsub models/ablation/cluster/submit_ablation_small.pbs
```

### Step 4: Run Analysis
```bash
# After all experiments complete
cd models/ablation
python run_updated_ablation.py --analysis-only
```

## 🔧 **Quick Fixes**

### Edit Existing Scripts
You can modify any script to reduce job count:

```bash
# Edit the array size in any .pbs file
#PBS -J 1-5    # Instead of 1-60

# Or edit the variants list to run fewer experiments
VARIANTS=("full_model" "no_evidential" "no_evt")  # Only 3 variants
```

### Check Queue Limits
```bash
# See your current limits
qstat -Q v1_gpu72

# Check how many jobs you have
qstat -u $USER | grep -c "Q\|R"
```

## 📊 **Experiment Priority**

If you can only run a few experiments, prioritize these:

### **High Priority (Core Ablations)**
1. `full_model` - Baseline with all components
2. `no_evidential` - Remove uncertainty quantification
3. `no_evt` - Remove extreme value theory
4. `mean_pool` - Remove attention mechanism

### **Medium Priority**
5. `cross_entropy` - Remove focal loss
6. `no_precursor` - Remove auxiliary head

### **Low Priority**
7. `fp32_training` - Mixed precision ablation
8. Sequence length variants

## 💡 **Tips**

- **Start with seed 0** for all variants first
- **Use smaller batch sizes** (5-10 jobs max)
- **Submit during off-peak hours** (evenings/weekends)
- **Monitor queue regularly** with `qstat -u $USER`
- **Check cluster status** before submitting large batches

## 🆘 **If Still Having Issues**

### Contact Cluster Support
```bash
# Check cluster documentation
module avail
qstat -Q

# Contact ICL support with your job requirements
```

### Run Locally Instead
```bash
# Run on your local machine with GPU
cd models/ablation
python run_updated_ablation.py --variants full_model no_evidential --seeds 0 --max-workers 1
```

---

**The small batch approach should work within your queue limits!** 🎯 