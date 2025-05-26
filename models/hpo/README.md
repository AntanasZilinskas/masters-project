# EVEREST HPO Framework

A comprehensive hyperparameter optimization framework for the EVEREST solar flare prediction model, implementing the exact three-tier Bayesian search protocol from academic research.

## Quick Start

```bash
# Single target optimization (local)
python run_hpo.py --target single --flare-class M --time-window 24 --max-trials 5

# Multi-target optimization (local)
python run_hpo.py --target multi --max-trials 10

# Imperial RCS Cluster (recommended for full study)
./cluster/submit_jobs.sh all
```

## Overview

This framework optimizes 6 hyperparameters across 9 target configurations (3 flare classes × 3 time windows) using a rigorous three-stage protocol:

- **Stage 1 (Exploration)**: 120 trials, 20 epochs each - broad parameter space exploration
- **Stage 2 (Refinement)**: 40 trials, 60 epochs each - focused search around promising regions  
- **Stage 3 (Confirmation)**: 6 trials, 120 epochs each - final validation of best configurations

**Total**: 166 trials per target × 9 targets = 1,494 trials for complete study

## Directory Structure

```
models/hpo/
├── __init__.py              # Package initialization
├── config.py                # Configuration management
├── objective.py             # Optuna objective function
├── study_manager.py         # Study orchestration
├── visualization.py         # Results visualization
├── run_hpo.py              # Main CLI interface
├── README.md               # This file
├── CLUSTER_SETUP_GUIDE.md  # Detailed cluster setup
└── cluster/                # Imperial RCS cluster scripts
    ├── README.md           # Cluster-specific documentation
    ├── submit_jobs.sh      # Job submission script
    ├── monitor_jobs.sh     # Job monitoring utilities
    ├── setup_environment.pbs
    ├── run_hpo_single.pbs
    ├── run_hpo_array.pbs
    └── run_hpo_cpu.pbs
```

## Dependencies

```bash
pip install optuna==3.6.1 ray[tune]==2.9.3 plotly==5.18.0 kaleido==0.2.1
```

## Usage

### Local Development

#### Single Target
```bash
python run_hpo.py --target single --flare-class M --time-window 24 --max-trials 166
```

#### Multiple Targets
```bash
python run_hpo.py --target multi --flare-classes C M X --time-windows 24 48 72 --max-trials 166
```

#### Configuration Options
```bash
python run_hpo.py \
    --target single \
    --flare-class M \
    --time-window 24 \
    --max-trials 166 \
    --timeout 86400 \
    --study-name "my_hpo_study" \
    --output-dir "results/custom_dir" \
    --n-jobs 4 \
    --resume
```

### Imperial RCS Cluster (Recommended)

For the complete study, use the cluster deployment:

```bash
# Complete workflow - setup + all 9 targets in parallel
./cluster/submit_jobs.sh all

# Individual commands
./cluster/submit_jobs.sh setup      # Environment setup
./cluster/submit_jobs.sh array      # All 9 targets
./cluster/submit_jobs.sh single M 24  # Specific target
./cluster/submit_jobs.sh cpu M 24     # CPU-only job

# Monitoring
./cluster/monitor_jobs.sh status    # Job status
./cluster/monitor_jobs.sh watch     # Real-time monitoring
./cluster/monitor_jobs.sh logs      # View recent logs
```

See [CLUSTER_SETUP_GUIDE.md](CLUSTER_SETUP_GUIDE.md) for detailed cluster instructions.

## Hyperparameter Search Space

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `embed_dim` | int | [64, 256] | 128 | Transformer embedding dimension |
| `num_blocks` | int | [4, 12] | 6 | Number of transformer blocks |
| `dropout` | float | [0.1, 0.5] | 0.2 | Dropout rate |
| `focal_gamma` | float | [0.0, 3.0] | 1.0 | Focal loss focusing parameter |
| `learning_rate` | float | [1e-5, 1e-2] | 3e-4 | AdamW learning rate |
| `batch_size` | categorical | [256, 512, 1024] | 512 | Training batch size |

## Target Configurations

| Configuration | Description | Expected TSS |
|---------------|-------------|--------------|
| C_24h | C-class flares, 24h prediction window | 0.65-0.75 |
| C_48h | C-class flares, 48h prediction window | 0.60-0.70 |
| C_72h | C-class flares, 72h prediction window | 0.55-0.65 |
| M_24h | M-class flares, 24h prediction window | 0.55-0.65 |
| M_48h | M-class flares, 48h prediction window | 0.50-0.60 |
| M_72h | M-class flares, 72h prediction window | 0.45-0.55 |
| X_24h | X-class flares, 24h prediction window | 0.45-0.55 |
| X_48h | X-class flares, 48h prediction window | 0.40-0.50 |
| X_72h | X-class flares, 72h prediction window | 0.35-0.45 |

## Results Analysis

### Output Structure
```
results/
├── hpo_C_24h/
│   ├── study.db                     # Optuna study database
│   ├── best_params.json             # Best hyperparameters
│   ├── optimization_history.png     # Trial performance over time
│   ├── parameter_importance.png     # Feature importance
│   ├── parallel_coordinates.png     # Parameter relationships
│   └── stage_analysis.png           # Three-stage breakdown
├── hpo_C_48h/
└── ... (9 directories total)
```

### Best Parameters Extraction
```python
from models.hpo import StudyManager

# Load study results
manager = StudyManager("results/hpo_M_24h/study.db")
best_params = manager.get_best_params()
best_value = manager.get_best_value()

print(f"Best TSS: {best_value:.4f}")
print(f"Best parameters: {best_params}")
```

### Cross-Target Analysis
```bash
python -c "
from models.hpo.visualization import create_cross_target_analysis
create_cross_target_analysis('results/', 'analysis/')
"
```

## Implementation Details

### Three-Stage Protocol
1. **Stage 1 - Exploration** (120 trials):
   - TPE sampler with n_startup_trials=20
   - MedianPruner for early stopping
   - 20 epochs per trial for quick evaluation

2. **Stage 2 - Refinement** (40 trials):
   - Continues from Stage 1 study
   - 60 epochs per trial for better estimates
   - Focused on promising parameter regions

3. **Stage 3 - Confirmation** (6 trials):
   - Top configurations from Stage 2
   - 120 epochs per trial for final validation
   - Comprehensive model evaluation

### Optimization Algorithm
- **Sampler**: TPESampler (Tree-structured Parzen Estimator)
- **Pruner**: MedianPruner with warmup_steps=5
- **Objective**: True Skill Statistic (TSS) maximization
- **Timeout**: 24 hours per target (cluster), 6 hours (local)

### Reproducibility
- Seeded random number generators
- Git commit tracking in study metadata
- Complete environment documentation
- Deterministic model initialization

## Performance Expectations

### Local Machine
- **Single trial**: 8-15 minutes (depends on hardware)
- **Single target (166 trials)**: 18-36 hours
- **Full study (1,494 trials)**: 7-14 days

### Imperial RCS Cluster (Recommended)
- **Single trial**: 10-15 minutes (L40S GPU)
- **Single target**: 24 hours (parallel trials)
- **Full study**: 24 hours (9 parallel array jobs)

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're in the project root directory
2. **Memory issues**: Reduce batch size or use smaller embed_dim
3. **CUDA errors**: Verify PyTorch CUDA installation
4. **Study not found**: Check output directory paths
5. **Slow performance**: Consider cluster deployment for large studies

### Debug Mode
```bash
python run_hpo.py --target single --flare-class M --time-window 24 --max-trials 1 --debug
```

### Logging
All optimization progress is logged to console and study database. Use `--verbose` for detailed output.

## API Reference

### Main Classes

#### `HPOObjective`
```python
from models.hpo import HPOObjective

objective = HPOObjective(
    flare_class="M",
    time_window=24,
    max_epochs=20,
    early_stopping_patience=5
)
```

#### `StudyManager`
```python
from models.hpo import StudyManager

manager = StudyManager(
    study_name="hpo_M_24h",
    storage="sqlite:///study.db",
    sampler_config={'n_startup_trials': 20}
)
```

#### `HPOVisualization`
```python
from models.hpo import HPOVisualization

viz = HPOVisualization(study_name="hpo_M_24h")
viz.create_optimization_history()
viz.create_parameter_importance()
```

### Configuration

Default configuration can be overridden:

```python
from models.hpo.config import HPOConfig

config = HPOConfig(
    max_trials=100,
    timeout=7200,
    n_jobs=2,
    optimization_direction="maximize"
)
```

## Contributing

1. Follow the existing code structure and documentation style
2. Add tests for new functionality in `tests/test_hpo/`
3. Update documentation for any API changes
4. Ensure compatibility with both local and cluster environments

## Citation

If you use this HPO framework in your research, please cite:

```bibtex
@misc{everest_hpo_2024,
  title={EVEREST Hyperparameter Optimization Framework},
  author={[Your Name]},
  year={2024},
  howpublished={GitHub repository},
  url={https://github.com/your-repo/masters-project}
}
``` 