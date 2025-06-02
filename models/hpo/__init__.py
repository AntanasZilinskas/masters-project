"""
Hyperparameter Optimization Package for EVEREST Solar Flare Prediction

This package implements a three-tier Bayesian search using Optuna v3.6
with Ray Tune orchestration for efficient hyperparameter optimization.
"""

__version__ = "1.0.0"

from .config import *
from .objective import HPOObjective
from .study_manager import StudyManager
from .visualization import HPOVisualizer

__all__ = [
    'HPOObjective',
    'StudyManager',
    'HPOVisualizer',
    'HPO_SEARCH_SPACE',
    'SEARCH_STAGES',
    'BEST_CONFIG',
    'EXPERIMENT_TARGETS',
    'OUTPUT_DIRS'
]
