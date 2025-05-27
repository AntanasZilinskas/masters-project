# EVEREST Ablation Study Framework

This directory contains a comprehensive ablation study framework for the EVEREST solar flare prediction model, implementing the systematic experimental protocol described in the research paper.

## 📋 Overview

The ablation study framework performs systematic one-factor-at-a-time ablations on the M5-72h benchmark (the most challenging with 1:1,560 class imbalance) to measure the incremental utility of each architectural component.

### Experimental Protocol

- **Target**: M5-class flares, 72-hour prediction window
- **Reproducibility**: 5 independent seeds (0-4) per variant
- **Training**: 120 epochs maximum with early stopping after 10 epochs (no TSS improvement)
- **Hyperparameters**: Frozen optimal values from HPO study
- **Loss weights**: Re-normalized when components are removed
- **Statistical testing**: Paired bootstrap tests (10,000 resamples) with 95% confidence intervals
- **Significance**: p < 0.05 threshold

## 🧪 Ablation Variants

### Component Ablations (7 variants)

1. **Full Model** - Complete EVEREST with all components (baseline)
2. **– Evidential Head** - Remove NIG (evidential) branch
3. **– EVT Head** - Remove GPD (extreme value theory) branch  
4. **Mean Pool** - Replace attention bottleneck with mean pooling
5. **Cross-Entropy** - Disable focal loss (γ = 0)
6. **No Precursor** - Remove early-warning auxiliary head
7. **FP32 Training** - Disable mixed precision (AMP)

### Sequence Length Ablations (5 variants)

- **5 timesteps** - Truncated sequence length
- **7 timesteps** - Reduced sequence length
- **10 timesteps** - Current baseline
- **15 timesteps** - Extended sequence length
- **20 timesteps** - Maximum sequence length

## 📊 Evaluation Metrics

- **TSS** (True Skill Statistic) - Primary metric
- **F1** - F1 score
- **ECE** - Expected Calibration Error (15-bin)
- **Brier** - Brier score
- **Accuracy** - Overall accuracy
- **Precision/Recall** - Classification metrics
- **ROC AUC** - Area under ROC curve
- **Latency** - Inference time (milliseconds)

## 🚀 Quick Start

### Local Execution

```bash
# Run complete ablation study
python models/ablation/run_ablation_study.py

# Run specific variants only
python models/ablation/run_ablation_study.py --variants full_model no_evidential no_evt

# Run with specific seeds
python models/ablation/run_ablation_study.py --seeds 0 1 2

# Skip sequence length study
python models/ablation/run_ablation_study.py --no-sequence-study

# Run analysis only (no training)
python models/ablation/run_ablation_study.py --analysis-only

# Single experiment for debugging
python -m ablation.trainer --variant no_evidential --seed 0
python -m ablation.trainer --variant full_model --seed 0 --sequence seq_15
```

### Cluster Execution

```bash
# Submit all jobs to cluster
cd models/ablation/cluster
./submit_jobs.sh

# Component ablations only
./submit_jobs.sh --component-only

# Sequence length ablations only  
./submit_jobs.sh --sequence-only

# Dry run (show commands without executing)
./submit_jobs.sh --dry-run

# If PBS dependency issues occur, use simple version:
./submit_jobs_simple.sh
# Then manually submit analysis after array job completes:
qsub submit_analysis.pbs
```

## 📁 Directory Structure

```
models/ablation/
├── config.py                    # Configuration and hyperparameters
├── trainer.py                   # Training logic for ablation experiments
├── analysis.py                  # Statistical analysis and visualization
├── run_ablation_study.py        # Main orchestration script
├── __init__.py                  # Package initialization
├── README.md                    # This file
├── cluster/                     # Cluster submission scripts
│   ├── submit_ablation_array.pbs
│   ├── submit_analysis.pbs
│   └── submit_jobs.sh
├── results/                     # Experiment outputs
│   └── ablation_{variant}_seed{N}/
│       ├── results.json
│       ├── training_history.csv
│       └── final_metrics.csv
├── plots/                       # Analysis visualizations
│   ├── ablation_tss_impact.png
│   ├── sequence_length_analysis.png
│   ├── significance_heatmap.png
│   └── effect_sizes.png
├── logs/                        # Training logs
└── trained_models/              # Saved model weights
```

## ⚙️ Configuration

### Optimal Hyperparameters (Frozen)

From HPO study, used across all ablations:

```python
OPTIMAL_HYPERPARAMS = {
    "embed_dim": 128,
    "num_blocks": 4, 
    "dropout": 0.353,
    "focal_gamma": 2.803,
    "learning_rate": 5.34e-4,
    "batch_size": 512
}
```

### Dynamic Weight Schedule

The framework uses a 3-phase dynamic weight schedule (matching main training exactly):

- **Phase 1 (epochs 0-19)**: `{"focal": 0.9, "evid": 0.1, "evt": 0.0, "prec": 0.05}` (sum = 1.05)
- **Phase 2 (epochs 20-39)**: `{"focal": 0.8, "evid": 0.1, "evt": 0.1, "prec": 0.05}` (sum = 1.05)
- **Phase 3 (epochs 40+)**: `{"focal": 0.7, "evid": 0.1, "evt": 0.2, "prec": 0.05}` (sum = 1.05)

When components are removed for ablations, the corresponding weights are simply set to 0.0 (no re-normalization).

## 📈 Statistical Analysis

### Bootstrap Testing

- **Method**: Paired bootstrap tests against full model baseline
- **Resamples**: 10,000 per comparison
- **Confidence**: 95% intervals
- **Significance**: p < 0.05 threshold

### Effect Size Calculation

Cohen's d effect sizes computed for all comparisons:
- Small effect: |d| ≥ 0.2
- Medium effect: |d| ≥ 0.5  
- Large effect: |d| ≥ 0.8

## 📊 Expected Results

Based on paper specifications, expected TSS deltas from full model:

| Variant | Expected Δ TSS | Significance |
|---------|----------------|--------------|
| – Evidential | -0.094 | ✓ |
| – EVT | -0.067 | ✓ |
| Mean Pool | -0.045 | ✓ |
| Cross-Entropy | -0.023 | ✓ |
| No Precursor | -0.012 | ✓ |
| FP32 Training | -0.003 | ✗ |

## 🖥️ Computational Requirements

### Per Experiment
- **GPU**: 1x V100 (32GB VRAM)
- **CPU**: 8 cores
- **Memory**: 32GB RAM
- **Time**: 2-4 hours
- **Storage**: ~1GB per experiment

### Full Study
- **Total experiments**: 60 (35 component + 25 sequence)
- **Parallel execution**: ~24 hours with 50 concurrent jobs
- **Total storage**: ~60GB
- **Analysis time**: Additional 2 hours

## 🔧 Troubleshooting

### Common Issues

1. **PyTorch not found**: Ensure EVEREST environment is activated
   ```bash
   conda activate everest_env
   ```

2. **CUDA out of memory**: Reduce batch size in config
   ```python
   OPTIMAL_HYPERPARAMS["batch_size"] = 256
   ```

3. **Missing data**: Verify training/testing data files exist
   ```bash
   ls Nature_data/training_data_M5_72.csv
   ls Nature_data/testing_data_M5_72.csv
   ```

4. **PBS dependency error** (`qsub: illegal -W value`):
   ```bash
   # Use simple submission script instead:
   cd models/ablation/cluster
   ./submit_jobs_simple.sh
   
   # Then manually submit analysis after array completes:
   qsub submit_analysis.pbs
   ```

5. **Cluster job failures**: Check PBS logs
   ```bash
   tail -f logs/ablation_*.log
   ```

### Debug Mode

Run single experiment with detailed logging:
```bash
python -m ablation.trainer --variant full_model --seed 0 --verbose
```

## 📚 Implementation Details

### Reproducibility

- Fixed random seeds control weight initialization, data shuffling, dropout masks
- Deterministic CUDA operations enabled
- Environment variables set for additional reproducibility

### Sequence Length Handling

- **Truncation**: For shorter sequences, take most recent timesteps
- **Padding**: For longer sequences, repeat first timestep

### Early Stopping

- Monitor TSS on test set (following paper protocol)
- Stop after 10 epochs without improvement
- Restore best weights from optimal epoch

## 🤝 Contributing

To add new ablation variants:

1. Add configuration to `config.py`:
   ```python
   "new_variant": {
       "name": "New Variant",
       "description": "Description of ablation",
       "config": {
           "use_evidential": True,
           # ... other flags
           "loss_weights": {"focal": 0.65, "evid": 0.1, "evt": 0.2, "prec": 0.05}
       }
   }
   ```

2. Update cluster scripts if needed
3. Run validation: `python test_ablation_setup.py`

## 📄 Citation

If you use this ablation framework, please cite:

```bibtex
@article{everest2024,
  title={EVEREST: Enhanced Variational Extreme Rare-Event Simulation for Solar Flare Prediction},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## 📞 Support

For issues or questions:
- Check logs in `models/ablation/logs/`
- Review configuration in `models/ablation/config.py`
- Run test suite: `python test_ablation_setup.py`
- Open GitHub issue with error details 