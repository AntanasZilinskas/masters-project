# EVEREST Ablation Study - Updated Hyperparameters Guide

## 🎯 Updated Optimal Hyperparameters

The ablation study has been updated with your new optimal hyperparameters from HPO:

```python
OPTIMAL_HYPERPARAMS = {
    "embed_dim": 64,           # Was: 128
    "num_blocks": 8,           # Was: 4  
    "dropout": 0.239,          # Was: 0.353
    "focal_gamma": 3.422,      # Was: 2.803
    "learning_rate": 0.000693, # Was: 0.000534
    "batch_size": 1024         # Was: 512
}
```

## 🚀 Best Script to Use

**Use `run_updated_ablation.py`** - This is the recommended script that:
- ✅ Uses the updated hyperparameters automatically
- ✅ Provides clear command-line options
- ✅ Handles all ablation variants and seeds
- ✅ Includes statistical analysis
- ✅ Shows progress and results

## 📋 Quick Commands

### 1. Run Complete Ablation Study (Recommended)
```bash
cd models/ablation
python run_updated_ablation.py
```

This runs:
- 7 component ablation variants × 5 seeds = 35 experiments
- 5 sequence length variants × 5 seeds = 25 experiments  
- **Total: 60 experiments**

### 2. Test with Subset (For Development)
```bash
# Test with just 2 seeds
python run_updated_ablation.py --seeds 0 1

# Test specific variants only
python run_updated_ablation.py --variants full_model no_evidential no_evt

# Skip sequence length study
python run_updated_ablation.py --no-sequence-study
```

### 3. Debug Mode (Single-threaded)
```bash
python run_updated_ablation.py --max-workers 1 --seeds 0
```

### 4. Analysis Only (If experiments already completed)
```bash
python run_updated_ablation.py --analysis-only
```

## 🖥️ Alternative Scripts (If Needed)

### Original Main Script
```bash
python run_ablation_study.py
```
- Uses the updated hyperparameters (config.py was updated)
- More basic interface
- Same functionality as run_updated_ablation.py

### Cluster Submission (For HPC)
```bash
cd cluster
./submit_jobs.sh                    # Full study
./submit_jobs.sh --component-only   # Component ablations only
./submit_jobs_simple.sh            # Simpler version
```

### Single Experiment (For Testing)
```bash
python -m ablation.trainer --variant no_evidential --seed 0
python -m ablation.trainer --variant full_model --seed 0 --sequence seq_15
```

## 📊 What Gets Generated

### Results Structure
```
models/ablation/
├── results/                     # Experiment outputs
│   ├── ablation_full_model_seed0/
│   ├── ablation_no_evidential_seed0/
│   ├── ablation_no_evt_seed0/
│   └── ...
├── plots/                       # Analysis visualizations
│   ├── ablation_tss_impact.png
│   ├── sequence_length_analysis.png
│   ├── significance_heatmap.png
│   └── effect_sizes.png
└── logs/                        # Training logs
```

### Key Output Files
- **`results.json`** - Final metrics for each experiment
- **`training_history.csv`** - Training curves
- **`ablation_summary.json`** - Statistical analysis
- **Plots** - Visualization of results and significance tests

## ⏱️ Expected Runtime

### Local Machine
- **Single experiment**: ~2-4 hours (depends on GPU)
- **Full study (60 experiments)**: ~120-240 hours sequential
- **Parallel (8 workers)**: ~15-30 hours

### Cluster
- **Full study**: ~4-8 hours with 50+ parallel jobs
- **Component only**: ~2-4 hours
- **Analysis**: Additional ~30 minutes

## 🎯 Recommended Workflow

### 1. Quick Test First
```bash
# Test with minimal setup
python run_updated_ablation.py --variants full_model no_evidential --seeds 0 --max-workers 1
```

### 2. Component Ablations
```bash
# Run all component ablations
python run_updated_ablation.py --no-sequence-study
```

### 3. Full Study
```bash
# Run everything
python run_updated_ablation.py
```

### 4. Analysis
Results are automatically analyzed, but you can re-run analysis:
```bash
python run_updated_ablation.py --analysis-only
```

## 🔧 Key Changes Made

1. **Updated `config.py`** with your optimal hyperparameters
2. **Created `run_updated_ablation.py`** for easy execution
3. **All variants automatically use new hyperparameters**
4. **Batch size increased to 1024** (may need more GPU memory)
5. **Model architecture updated** (embed_dim=64, num_blocks=8)

## 💡 Tips

- **Start small**: Test with 1-2 variants and 1 seed first
- **Monitor GPU memory**: Batch size 1024 needs more VRAM
- **Use cluster**: For full study, cluster execution is much faster
- **Check logs**: Monitor training progress in `logs/` directory
- **Validate results**: Compare with expected effect sizes in README.md

## 🆘 Troubleshooting

### GPU Memory Issues
```bash
# Reduce batch size temporarily in config.py if needed
# Or run with smaller variants first
python run_updated_ablation.py --variants no_precursor fp32_training
```

### Import Errors
```bash
# Make sure you're in the right directory
cd models/ablation
python run_updated_ablation.py
```

### Slow Training
```bash
# Use fewer seeds for testing
python run_updated_ablation.py --seeds 0 1 2
```

---

**Your ablation study is now ready with the updated optimal hyperparameters!** 🎉 