"""
Configuration for EVEREST Hyperparameter Optimization Study

This module defines the search space, optimization stages, and hyperparameter
priors as described in the research paper.
"""

import os
from typing import Dict, Any, List, Tuple


# ============================================================================
# Search Space Configuration (Table from paper)
# ============================================================================

HPO_SEARCH_SPACE = {
    # Model capacity
    "embed_dim": [64, 128, 192, 256],  # capacity vs. latency
    "num_blocks": [4, 6, 8],           # encoder depth L - receptive field
    
    # Regularization
    "dropout": (0.05, 0.40),           # over-fit control - Uniform[0.05,0.40]
    
    # Class-imbalance focus  
    "focal_gamma": (1.0, 4.0),         # minority gradient - Uniform[1,4]
    
    # Optimizer dynamics
    "learning_rate": (2e-4, 8e-4),     # peak LR - Log-Uniform[2×10^-4, 8×10^-4]
    
    # Throughput knob
    "batch_size": [256, 512, 768, 1024]  # throughput vs. generalisation
}

# Best configuration found (from paper)
BEST_CONFIG = {
    "embed_dim": 128,
    "num_blocks": 6,
    "dropout": 0.20,
    "focal_gamma": 2.0,
    "learning_rate": 4e-4,
    "batch_size": 512
}

# ============================================================================
# Three-Stage Search Protocol (Table from paper)  
# ============================================================================

SEARCH_STAGES = {
    "exploration": {
        "trials": 120,
        "epochs": 20,
        "purpose": "Coarse global sweep"
    },
    "refinement": {
        "trials": 40, 
        "epochs": 60,
        "purpose": "Zoom on top quartile"
    },
    "confirmation": {
        "trials": 6,
        "epochs": 120,
        "purpose": "Full-length convergence"
    }
}

# ============================================================================
# Fixed Model Architecture 
# ============================================================================

FIXED_ARCHITECTURE = {
    "input_shape": (10, 9),    # (timesteps, features)
    "ff_dim": 256,             # feedforward dimension
    "num_heads": 4,            # attention heads
    
    # Ablation flags (set based on study requirements)
    "use_attention_bottleneck": True,
    "use_evidential": True,
    "use_evt": True,
    "use_precursor": True
}

# ============================================================================
# Optimization Configuration
# ============================================================================

OPTUNA_CONFIG = {
    "study_name": "everest_hpo_v4.1",
    "direction": "maximize",     # maximize TSS
    "sampler": "TPESampler",     # Tree-structured Parzen Estimator  
    "pruner": "MedianPruner",    # median-stopping pruner
    "storage": "sqlite:///models/hpo/optuna_studies.db",
    
    # Pruner configuration
    "pruner_config": {
        "n_startup_trials": 10,     # min trials before pruning
        "n_warmup_steps": 5,        # min epochs before pruning
        "interval_steps": 1         # check every epoch
    }
}

RAY_TUNE_CONFIG = {
    "num_cpus": os.cpu_count(),
    "num_gpus": 1 if os.getenv("CUDA_VISIBLE_DEVICES") else 0,
    "max_concurrent_trials": 4,  # adjust based on available resources
    "grace_period": 10,          # minimum epochs before stopping
    "reduction_factor": 2        # factor for successive halving
}

# ============================================================================
# Experiment Configuration
# ============================================================================

# Target configurations to optimize
EXPERIMENT_TARGETS = [
    {"flare_class": "C", "time_window": "24"},
    {"flare_class": "M", "time_window": "24"},
    {"flare_class": "M5", "time_window": "24"},
    {"flare_class": "C", "time_window": "48"},
    {"flare_class": "M", "time_window": "48"},
    {"flare_class": "M5", "time_window": "48"},
    {"flare_class": "C", "time_window": "72"},
    {"flare_class": "M", "time_window": "72"},
    {"flare_class": "M5", "time_window": "72"}
]

# Default target for single experiments
DEFAULT_TARGET = {"flare_class": "M", "time_window": "24"}

# ============================================================================
# Reproducibility & Logging
# ============================================================================

REPRODUCIBILITY_CONFIG = {
    "git_tag": "v4.1-hpo-prod",
    "random_seed": 42,
    "torch_deterministic": True,
    "log_level": "INFO"
}

# Output directories
OUTPUT_DIRS = {
    "base": "models/hpo",
    "studies": "models/hpo/studies", 
    "results": "models/hpo/results",
    "logs": "models/hpo/logs",
    "plots": "models/hpo/plots",
    "models": "models/hpo/best_models"
}

# ============================================================================
# Evaluation Metrics
# ============================================================================

PRIMARY_METRIC = "TSS"  # True Skill Statistic - main optimization target

SECONDARY_METRICS = [
    "accuracy",
    "precision", 
    "recall",
    "roc_auc",
    "brier_score",
    "ece",  # Expected Calibration Error
    "inference_latency"
]

# Performance thresholds (for early stopping/filtering)
PERFORMANCE_THRESHOLDS = {
    "min_tss": 0.3,      # minimum acceptable TSS
    "max_latency": 60.0,  # maximum acceptable inference time (seconds)
    "min_accuracy": 0.7   # minimum acceptable accuracy
}

# ============================================================================
# Loss Configuration 
# ============================================================================

LOSS_WEIGHTS_CONFIG = {
    # 3-phase schedule as in the model
    "phase_1": {"focal": 0.9, "evid": 0.1, "evt": 0.0, "prec": 0.05},  # epochs 0-19
    "phase_2": {"focal": 0.8, "evid": 0.1, "evt": 0.1, "prec": 0.05},  # epochs 20-39  
    "phase_3": {"focal": 0.7, "evid": 0.1, "evt": 0.2, "prec": 0.05}   # epochs 40+
}

# ============================================================================
# Utility Functions
# ============================================================================

def create_output_dirs() -> None:
    """Create all necessary output directories."""
    for dir_path in OUTPUT_DIRS.values():
        os.makedirs(dir_path, exist_ok=True)

def get_stage_config(stage_name: str) -> Dict[str, Any]:
    """Get configuration for a specific optimization stage."""
    if stage_name not in SEARCH_STAGES:
        raise ValueError(f"Unknown stage: {stage_name}. Choose from {list(SEARCH_STAGES.keys())}")
    return SEARCH_STAGES[stage_name]

def get_total_trials() -> int:
    """Get total number of trials across all stages."""
    return sum(stage["trials"] for stage in SEARCH_STAGES.values())

def get_search_space_size() -> int:
    """Estimate the discrete search space size."""
    size = 1
    for param, values in HPO_SEARCH_SPACE.items():
        if isinstance(values, list):
            size *= len(values)
        else:  # continuous parameter
            size *= 100  # rough approximation
    return size

def validate_config() -> bool:
    """Validate the configuration for consistency."""
    # Check that all required directories can be created
    try:
        create_output_dirs()
    except Exception as e:
        print(f"Error creating directories: {e}")
        return False
    
    # Test database write permissions
    try:
        import sqlite3
        db_path = OPTUNA_CONFIG["storage"].replace("sqlite:///", "")
        
        # Create parent directory if needed
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Test database connection and write
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS test_table (id INTEGER)")
        conn.execute("DROP TABLE IF EXISTS test_table")
        conn.close()
        print(f"✅ Database write permissions validated: {db_path}")
        
    except Exception as e:
        print(f"❌ Database validation failed: {e}")
        return False
    
    # Check that best config is within search space
    for param, value in BEST_CONFIG.items():
        if param not in HPO_SEARCH_SPACE:
            print(f"Best config parameter {param} not in search space")
            return False
            
        space_values = HPO_SEARCH_SPACE[param]
        if isinstance(space_values, list):
            if value not in space_values:
                print(f"Best config value {value} not in search space {space_values}")
                return False
        else:  # tuple (min, max)
            if not (space_values[0] <= value <= space_values[1]):
                print(f"Best config value {value} not in range {space_values}")
                return False
    
    return True

if __name__ == "__main__":
    # Validate configuration when run directly
    print("Validating HPO configuration...")
    if validate_config():
        print("✅ Configuration validated successfully")
        print(f"Total trials across all stages: {get_total_trials()}")
        print(f"Estimated search space size: {get_search_space_size():,}")
    else:
        print("❌ Configuration validation failed") 