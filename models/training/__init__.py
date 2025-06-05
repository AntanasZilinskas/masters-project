"""
EVEREST Production Training Package

This package provides comprehensive production training capabilities for EVEREST
solar flare prediction models with threshold optimization and statistical analysis.

Key Components:
- config.py: Training configuration and hyperparameters
- trainer.py: Production model trainer with threshold optimization
- run_production_training.py: Orchestration script for all experiments
- analysis.py: Statistical analysis and visualization
- cluster/: PBS job submission scripts

Usage:
    # Train single model
    python -m training.trainer --flare_class M5 --time_window 72 --seed 0

    # Run all experiments
    python models/training/run_production_training.py --mode all

    # Cluster submission
    cd models/training/cluster && ./submit_jobs.sh
"""

from .config import (
    TRAINING_TARGETS,
    RANDOM_SEEDS,
    FIXED_ARCHITECTURE,
    TRAINING_HYPERPARAMS,
    THRESHOLD_CONFIG,
    EVALUATION_METRICS,
    get_experiment_name,
    get_all_experiments,
)

from .trainer import ProductionTrainer, train_production_model
from .analysis import ProductionAnalyzer

__version__ = "4.1.0"
__author__ = "EVEREST Team"

__all__ = [
    "TRAINING_TARGETS",
    "RANDOM_SEEDS",
    "FIXED_ARCHITECTURE",
    "TRAINING_HYPERPARAMS",
    "THRESHOLD_CONFIG",
    "EVALUATION_METRICS",
    "get_experiment_name",
    "get_all_experiments",
    "ProductionTrainer",
    "train_production_model",
    "ProductionAnalyzer",
]
