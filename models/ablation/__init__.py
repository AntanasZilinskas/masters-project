"""
EVEREST Ablation Studies Package

This package implements systematic ablation studies for the EVEREST solar flare
prediction model, following the experimental protocol described in the paper.

Modules:
    config: Configuration and hyperparameters for ablation studies
    trainer: Training logic for individual ablation experiments
    analysis: Statistical analysis and visualization
    run_ablation_study: Main orchestration script

Usage:
    # Run complete ablation study
    python -m ablation.run_ablation_study

    # Run single experiment
    python -m ablation.trainer --variant no_evidential --seed 0

    # Run analysis only
    python -m ablation.run_ablation_study --analysis-only
"""

from .config import (
    ABLATION_VARIANTS,
    SEQUENCE_LENGTH_VARIANTS,
    RANDOM_SEEDS,
    PRIMARY_TARGET,
    EVALUATION_METRICS,
    get_variant_config,
    get_sequence_config,
    validate_config
)

# Conditional imports to avoid PyTorch dependency for config-only usage
try:
    from .trainer import AblationTrainer, train_ablation_variant
    from .analysis import AblationAnalyzer
    _TRAINING_AVAILABLE = True
except ImportError:
    _TRAINING_AVAILABLE = False
    AblationTrainer = None
    train_ablation_variant = None
    AblationAnalyzer = None

__version__ = "1.0.0"
__author__ = "EVEREST Team"

__all__ = [
    # Configuration
    "ABLATION_VARIANTS",
    "SEQUENCE_LENGTH_VARIANTS",
    "RANDOM_SEEDS",
    "PRIMARY_TARGET",
    "EVALUATION_METRICS",
    "get_variant_config",
    "get_sequence_config",
    "validate_config",

    # Training
    "AblationTrainer",
    "train_ablation_variant",

    # Analysis
    "AblationAnalyzer"
]
