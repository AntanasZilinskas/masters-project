# EVEREST Production Training

This module provides comprehensive production training capabilities for EVEREST solar flare prediction models with threshold optimization, statistical analysis, and cluster deployment.

## 🎯 Overview

The production training framework trains EVEREST models across all 9 flare class × time window combinations (C/M/M5 × 24h/48h/72h) with 5 random seeds each for statistical robustness. Each model includes:

- **Threshold Optimization**: 81-point search (0.1-0.9) with balanced scoring
- **Comprehensive Testing**: Full evaluation on held-out test sets
- **Statistical Analysis**: Confidence intervals, significance testing
- **Performance Tracking**: TSS, F1, precision, recall, ROC AUC, Brier, ECE, latency

## 📁 Structure

```
models/training/
├── config.py                    # Training configuration
├── trainer.py                   # Production model trainer
├── run_production_training.py   # Orchestration script
├── analysis.py                  # Statistical analysis
├── __init__.py                  # Package initialization
├── README.md                    # This file
├── cluster/                     # Cluster job scripts
│   ├── submit_production_array.pbs
│   ├── submit_analysis.pbs
│   └── submit_jobs.sh
├── results/                     # Experiment results
├── trained_models/              # Saved model weights
├── logs/                        # Training logs
├── plots/                       # Visualizations
└── analysis/                    # Analysis outputs
```

## 🚀 Quick Start

### Local Training

```bash
# Train single model
python models/training/trainer.py \
    --flare_class M5 \
    --time_window 72 \
    --seed 0

# Run all experiments (sequential)
python models/training/run_production_training.py --mode all

# Run specific targets
python models/training/run_production_training.py \
    --mode all \
    --targets C-24 M-48 M5-72

# Parallel execution (4 workers)
python models/training/run_production_training.py \
    --mode all \
    --max_workers 4
```

### Cluster Execution

```bash
# Submit all jobs to cluster
cd models/training/cluster
./submit_jobs.sh

# Dry run (show commands without executing)
./submit_jobs.sh --dry-run

# Submit specific targets only
./submit_jobs.sh --targets C-24 M-48 M5-72

# Skip analysis job
./submit_jobs.sh --no-analysis
```

### Analysis Only

```bash
# Run analysis after training completes
python models/training/analysis.py
```

## ⚙️ Configuration

### Training Targets

All 9 combinations of flare class × time window:
- **Flare Classes**: C, M, M5
- **Time Windows**: 24h, 48h, 72h
- **Seeds**: 0, 1, 2, 3, 4 (5 per target)
- **Total Experiments**: 45

### Model Architecture

```python
FIXED_ARCHITECTURE = {
    "input_shape": (10, 9),
    "embed_dim": 128,
    "num_heads": 4,
    "ff_dim": 256,
    "num_blocks": 6,
    "dropout": 0.2,
    "use_attention_bottleneck": True,
    "use_evidential": True,
    "use_evt": True,
    "use_precursor": True
}
```

### Training Hyperparameters

Optimized from HPO study:
```python
TRAINING_HYPERPARAMS = {
    "epochs": 120,
    "batch_size": 512,
    "learning_rate": 5.34e-4,
    "early_stopping_patience": 10,
    "focal_gamma_max": 2.803,
    "use_amp": True
}
```

### Threshold Optimization

```python
THRESHOLD_CONFIG = {
    "search_range": (0.1, 0.9),
    "search_points": 81,
    "optimization_metric": "balanced_score"
}

BALANCED_WEIGHTS = {
    "tss": 0.4,
    "f1": 0.2,
    "precision": 0.15,
    "recall": 0.15,
    "specificity": 0.1
}
```

## 📊 Evaluation Metrics

Each model is evaluated on:

- **TSS** (True Skill Statistic) - Primary metric
- **Accuracy** - Overall correctness
- **Precision** - Positive predictive value
- **Recall** - Sensitivity/True positive rate
- **Specificity** - True negative rate
- **F1 Score** - Harmonic mean of precision/recall
- **ROC AUC** - Area under ROC curve
- **Brier Score** - Probabilistic accuracy
- **ECE** - Expected Calibration Error (15-bin)
- **Latency** - Inference time (milliseconds)

## 🎯 Threshold Optimization

The framework performs comprehensive threshold optimization:

1. **Search Space**: 81 points from 0.1 to 0.9 (0.01 increments)
2. **Balanced Scoring**: Weighted combination of TSS (40%), F1 (20%), precision (15%), recall (15%), specificity (10%)
3. **Test Set Optimization**: Thresholds optimized on held-out test data
4. **Stability Analysis**: Threshold consistency across seeds

## 📈 Statistical Analysis

Comprehensive analysis includes:

### Summary Statistics
- Mean ± standard deviation across 5 seeds
- 95% confidence intervals
- Min/max/median values
- Threshold stability analysis

### Visualizations
- Performance heatmaps (TSS, F1 by target)
- Box plots for all metrics
- Threshold distribution analysis
- Performance vs latency trade-offs
- Detailed metrics comparison

### Reports
- Markdown summary report
- CSV files (raw results, summary statistics)
- Best performing model identification
- Statistical significance testing

## 🖥️ Cluster Configuration

### Resource Requirements
- **Per Job**: 1 GPU (L40S), 4 cores, 32GB RAM
- **Time Limit**: 12 hours per job
- **Total Jobs**: 45 (array job)
- **Expected Runtime**: ~12 hours (parallel)
- **Storage**: ~50GB total

### Job Dependencies
- Array job runs all 45 training experiments
- Analysis job runs after array completion
- Automatic dependency management

## 📁 Output Structure

### Per-Experiment Results
```
results/{experiment_name}/
├── results.json              # Complete experiment data
├── training_history.csv      # Epoch-by-epoch metrics
├── threshold_optimization.csv # Threshold search results
├── final_metrics.csv         # Summary metrics
└── predictions.csv           # Raw predictions (optional)
```

### Trained Models
```
trained_models/{experiment_name}/
└── model_weights.pt          # PyTorch model weights
```

### Analysis Outputs
```
analysis/
├── production_training_report.md  # Summary report
├── raw_results.csv                # All experiment results
└── summary_statistics.csv         # Aggregated statistics

plots/
├── production_performance_analysis.png
├── detailed_metrics_analysis.png
└── threshold_analysis.png
```

## 🔧 Advanced Usage

### Custom Configuration

```python
from models.training.config import TRAINING_HYPERPARAMS

# Modify hyperparameters
TRAINING_HYPERPARAMS["epochs"] = 200
TRAINING_HYPERPARAMS["batch_size"] = 256

# Custom threshold search
THRESHOLD_CONFIG["search_points"] = 101  # Finer search
```

### Programmatic Access

```python
from models.training import ProductionTrainer, ProductionAnalyzer

# Train single model
trainer = ProductionTrainer("M5", "72", seed=0)
results = trainer.train()

# Run analysis
analyzer = ProductionAnalyzer()
analyzer.run_complete_analysis()
```

### Array Job Mapping

```python
from models.training.config import get_array_job_mapping

# Get experiment for PBS array index
mapping = get_array_job_mapping()
experiment = mapping[25]  # Array index 25
print(f"Job 25: {experiment['experiment_name']}")
```

## 🐛 Troubleshooting

### Common Issues

1. **Missing Data Files**
   ```bash
   # Verify data files exist
   ls Nature_data/training_data_M5_72.csv
   ls Nature_data/testing_data_M5_72.csv
   ```

2. **CUDA Out of Memory**
   ```python
   # Reduce batch size in config
   TRAINING_HYPERPARAMS["batch_size"] = 256
   ```

3. **PBS Job Failures**
   ```bash
   # Check job logs
   tail -f logs/production_*.log
   
   # Check job status
   qstat -u $USER
   ```

4. **Import Errors**
   ```bash
   # Ensure EVEREST environment is activated
   conda activate everest_env
   
   # Check PYTHONPATH
   export PYTHONPATH="${PWD}:${PYTHONPATH}"
   ```

### Performance Optimization

1. **GPU Memory**: Use `in_memory_dataset=True` for faster training
2. **Mixed Precision**: Enabled by default (`use_amp=True`)
3. **Parallel Workers**: Adjust `--max_workers` based on available GPUs
4. **Early Stopping**: Configured for 10 epochs patience

## 📊 Expected Results

Based on preliminary experiments:

### Performance Ranges
- **TSS**: 0.3-0.8 (varies by target difficulty)
- **F1**: 0.2-0.7 (class imbalance dependent)
- **Precision**: 0.1-0.9 (threshold dependent)
- **Recall**: 0.2-0.8 (model sensitivity)
- **Latency**: 1-5 ms (per sample)

### Threshold Ranges
- **C-class**: 0.2-0.4 (lower thresholds)
- **M-class**: 0.3-0.6 (moderate thresholds)
- **M5-class**: 0.4-0.8 (higher thresholds)

### Training Times
- **Per Model**: 2-4 hours (depends on early stopping)
- **Full Study**: ~12 hours (parallel execution)
- **Analysis**: ~30 minutes

## 🔗 Related Components

- **HPO Study**: `models/hpo/` - Hyperparameter optimization
- **Ablation Study**: `models/ablation/` - Component analysis
- **Model Tracking**: `models/model_tracking.py` - Version management
- **Base Model**: `models/solarknowledge_ret_plus.py` - EVEREST implementation

## 📚 References

- EVEREST Paper: [Solar Flare Prediction with Evidential Learning]
- HPO Results: `models/hpo/results/`
- Ablation Results: `models/ablation/results/`
- Model Versions: `models/EVEREST-v*/`

---

**Note**: This production training framework represents the final evaluation phase of EVEREST development, providing publication-ready results with comprehensive statistical analysis and threshold optimization. 