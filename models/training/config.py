"""
EVEREST Production Model Training Configuration

This module defines the configuration for training production EVEREST models
across all flare class × time window combinations with multiple seeds for
statistical analysis and threshold optimization.
"""

import os
from typing import Dict, List, Any, Tuple

# ============================================================================
# TRAINING TARGETS
# ============================================================================

# All 9 combinations of flare class × time window
TRAINING_TARGETS = [
    {"flare_class": "C", "time_window": "24"},
    {"flare_class": "C", "time_window": "48"},
    {"flare_class": "C", "time_window": "72"},
    {"flare_class": "M", "time_window": "24"},
    {"flare_class": "M", "time_window": "48"},
    {"flare_class": "M", "time_window": "72"},
    {"flare_class": "M5", "time_window": "24"},
    {"flare_class": "M5", "time_window": "48"},
    {"flare_class": "M5", "time_window": "72"}
]

# Random seeds for statistical analysis (5 runs per target)
RANDOM_SEEDS = [0, 1, 2, 3, 4]

# Total experiments: 9 targets × 5 seeds = 45 experiments
TOTAL_EXPERIMENTS = len(TRAINING_TARGETS) * len(RANDOM_SEEDS)

# ============================================================================
# MODEL ARCHITECTURE & HYPERPARAMETERS
# ============================================================================

# Fixed architecture parameters
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

# Training hyperparameters (from HPO results)
TRAINING_HYPERPARAMS = {
    "epochs": 300,
    "batch_size": 512,
    "learning_rate": 5.34e-4,
    "weight_decay": 1e-4,
    "early_stopping_patience": 10,
    "focal_gamma_max": 2.803,
    "warmup_epochs": 50,
    "use_amp": True,  # Mixed precision training
    "in_memory_dataset": True  # GPU memory optimization
}

# Dynamic loss weight schedule (3-phase)
LOSS_WEIGHT_SCHEDULE = {
    "phase_1": {"focal": 0.9, "evid": 0.1, "evt": 0.0, "prec": 0.05},  # epochs 0-19
    "phase_2": {"focal": 0.8, "evid": 0.1, "evt": 0.1, "prec": 0.05},  # epochs 20-39
    "phase_3": {"focal": 0.7, "evid": 0.1, "evt": 0.2, "prec": 0.05}   # epochs 40+
}

# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

# Threshold search configuration
THRESHOLD_CONFIG = {
    "search_range": (0.1, 0.9),     # Search between 10% and 90%
    "search_points": 81,            # 0.1, 0.11, 0.12, ..., 0.9 (0.01 steps)
    "optimization_metric": "balanced_score",  # Custom balanced metric
    "fallback_threshold": 0.5       # Default if optimization fails
}

# Balanced scoring weights for threshold optimization
BALANCED_WEIGHTS = {
    "tss": 0.4,        # True Skill Statistic (primary)
    "f1": 0.2,         # F1 score
    "precision": 0.15,  # Precision
    "recall": 0.15,    # Recall/Sensitivity
    "specificity": 0.1  # Specificity
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================

EVALUATION_METRICS = [
    "tss",           # True Skill Statistic (primary)
    "accuracy",      # Overall accuracy
    "precision",     # Precision
    "recall",        # Recall (sensitivity)
    "specificity",   # Specificity
    "f1",            # F1 score
    "roc_auc",       # ROC AUC
    "brier",         # Brier score
    "ece",           # Expected Calibration Error (15-bin)
    "latency_ms"     # Inference latency (milliseconds)
]

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

OUTPUT_CONFIG = {
    "base_dir": "models/training",
    "results_dir": "models/training/results",
    "models_dir": "models/training/trained_models",
    "logs_dir": "models/training/logs",
    "plots_dir": "models/training/plots",
    "analysis_dir": "models/training/analysis",
    "git_tag": "v4.1-production",
    "save_raw_predictions": True,
    "save_threshold_curves": True,
    "save_model_artifacts": True
}

# ============================================================================
# CLUSTER CONFIGURATION
# ============================================================================

CLUSTER_CONFIG = {
    "job_time_limit": "24:00:00",  # 24 hours per job
    "memory_per_job": "24gb",
    "gpu_type": "Any available GPU",
    "cpus_per_task": 4,
    "array_job_limit": 20,  # Max concurrent jobs
    "partition": "gpu"
}

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

STATISTICAL_CONFIG = {
    "confidence_level": 0.95,       # 95% confidence intervals
    "bootstrap_samples": 10000,     # Bootstrap resamples for CI
    "significance_threshold": 0.05,  # p < 0.05 for significance
    "effect_size_threshold": 0.02   # Minimum meaningful TSS difference
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_experiment_name(flare_class: str, time_window: str, seed: int) -> str:
    """Generate standardized experiment name."""
    return f"everest_{flare_class}_{time_window}h_seed{seed}"


def get_all_experiments() -> List[Dict[str, Any]]:
    """Get list of all experiment configurations."""
    experiments = []
    for target in TRAINING_TARGETS:
        for seed in RANDOM_SEEDS:
            experiments.append({
                "flare_class": target["flare_class"],
                "time_window": target["time_window"],
                "seed": seed,
                "experiment_name": get_experiment_name(
                    target["flare_class"], target["time_window"], seed
                )
            })
    return experiments


def get_experiments_by_target(flare_class: str, time_window: str) -> List[Dict[str, Any]]:
    """Get all experiments for a specific target."""
    return [
        exp for exp in get_all_experiments()
        if exp["flare_class"] == flare_class and exp["time_window"] == time_window
    ]


def get_threshold_search_points() -> List[float]:
    """Get threshold values to search over."""
    start, end = THRESHOLD_CONFIG["search_range"]
    n_points = THRESHOLD_CONFIG["search_points"]
    return [start + i * (end - start) / (n_points - 1) for i in range(n_points)]


def calculate_balanced_score(metrics: Dict[str, float]) -> float:
    """Calculate balanced score for threshold optimization."""
    score = 0.0
    for metric, weight in BALANCED_WEIGHTS.items():
        if metric in metrics:
            score += weight * metrics[metric]
    return score


def create_output_directories():
    """Create all necessary output directories."""
    for dir_path in OUTPUT_CONFIG.values():
        if isinstance(dir_path, str) and ("/" in dir_path):
            os.makedirs(dir_path, exist_ok=True)


def validate_config():
    """Validate configuration consistency."""
    # Check that weights sum to 1.0
    total_weight = sum(BALANCED_WEIGHTS.values())
    if abs(total_weight - 1.0) > 1e-6:
        print(f"Warning: Balanced weights sum to {total_weight:.6f}, not 1.0")

    # Check that all required directories can be created
    create_output_directories()

    # Validate threshold configuration
    start, end = THRESHOLD_CONFIG["search_range"]
    if not (0.0 <= start < end <= 1.0):
        raise ValueError(f"Invalid threshold range: {THRESHOLD_CONFIG['search_range']}")

    print("✅ Production training configuration validated successfully")


def get_array_job_mapping() -> Dict[int, Dict[str, Any]]:
    """Get mapping from array job index to experiment configuration."""
    experiments = get_all_experiments()
    return {i + 1: exp for i, exp in enumerate(experiments)}  # PBS arrays start at 1


if __name__ == "__main__":
    validate_config()
    print(f"📊 Configured {len(TRAINING_TARGETS)} training targets")
    print(f"🎲 Using {len(RANDOM_SEEDS)} random seeds")
    print(f"🔬 Total experiments: {TOTAL_EXPERIMENTS}")
    print(f"🎯 Threshold search points: {THRESHOLD_CONFIG['search_points']}")

    # Show experiment mapping
    mapping = get_array_job_mapping()
    print(f"\n📋 Array job mapping (first 5):")
    for i in range(1, min(6, len(mapping) + 1)):
        exp = mapping[i]
        print(f"   Job {i}: {exp['experiment_name']}")
