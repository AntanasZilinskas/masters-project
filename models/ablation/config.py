"""
Configuration for EVEREST Ablation Studies

This module defines all configurations for systematic ablation studies
following the experimental protocol described in the paper.
"""

import os
from typing import Dict, List, Any, Optional

# ============================================================================
# OPTIMAL HYPERPARAMETERS (from solarknowledge_ret_plus.py - frozen for all ablations)
# ============================================================================
OPTIMAL_HYPERPARAMS = {
    "embed_dim": 128,          # Updated from 64 to match solarknowledge_ret_plus.py default
    "num_blocks": 6,           # Updated from 8 to match solarknowledge_ret_plus.py default
    "dropout": 0.2,            # Updated from 0.23876978467047777 to match solarknowledge_ret_plus.py default
    "focal_gamma": 2.0,        # Updated from 3.4223204654921875 to match gamma_max from solarknowledge_ret_plus.py
    "learning_rate": 3e-4,     # Updated from 0.0006926769179941219 to match solarknowledge_ret_plus.py default
    "batch_size": 512          # Updated from 1024 to match solarknowledge_ret_plus.py default
}

# Fixed architecture settings
FIXED_ARCHITECTURE = {
    "input_shape": (10, 9),  # Will be varied for sequence length study
    "num_heads": 4,
    "ff_dim": 256,
    "early_stopping_patience": 10,  # Paper specifies 10 epochs for ablation
    "max_epochs": 300        # Updated from 120 to match notebook (300 epochs)
}

# ============================================================================
# ABLATION STUDY CONFIGURATION
# ============================================================================

# Primary target for ablation (most challenging benchmark)
PRIMARY_TARGET = {
    "flare_class": "M5",
    "time_window": "72"
}

# ALL TARGET COMBINATIONS for comprehensive ablation analysis
# (Enable for full cross-task comparison - currently focused on primary target for efficiency)
ALL_TARGETS = [
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

# Control flag for comprehensive vs focused ablation
COMPREHENSIVE_ABLATION = False  # Set to True to run across all 9 targets

# Random seeds for reproducibility (5 independent runs)
RANDOM_SEEDS = [0, 1, 2, 3, 4]

# Training configuration
TRAINING_CONFIG = {
    "epochs": 300,             # Updated from 120 to match notebook (300 epochs)
    "early_stopping_patience": 10,
    "track_emissions": False,  # Disable for speed in ablation studies
    "in_memory_dataset": True, # Updated to match notebook setting
    "use_amp": True,  # Mixed precision (will be ablated)
}

# Dynamic 3-phase weight schedule (matches main training exactly)
# Note: These weights sum to 1.05, which is intentional and matches your training
DYNAMIC_WEIGHT_SCHEDULE = {
    "phase_1": {"epochs": "0-19", "weights": {"focal": 0.9, "evid": 0.1, "evt": 0.0, "prec": 0.05}},
    "phase_2": {"epochs": "20-39", "weights": {"focal": 0.8, "evid": 0.1, "evt": 0.1, "prec": 0.05}},
    "phase_3": {"epochs": "40+", "weights": {"focal": 0.7, "evid": 0.1, "evt": 0.2, "prec": 0.05}}
}

# ============================================================================
# ABLATION VARIANTS
# ============================================================================

ABLATION_VARIANTS = {
    "full_model": {
        "name": "Full Model (Baseline)",
        "description": "Complete EVEREST model with all components",
        "config": {
            "use_evidential": True,
            "use_evt": True,
            "use_attention_bottleneck": True,
            "use_precursor": True,
            "focal_gamma": OPTIMAL_HYPERPARAMS["focal_gamma"],
            "use_amp": True,
            # Note: Uses dynamic 3-phase schedule in training (see trainer.py)
            "loss_weights": {"focal": 0.7, "evid": 0.1, "evt": 0.2, "prec": 0.05}
        }
    },
    
    "no_evidential": {
        "name": "– Evidential Head",
        "description": "Remove NIG (evidential) branch",
        "config": {
            "use_evidential": False,
            "use_evt": True,
            "use_attention_bottleneck": True,
            "use_precursor": True,
            "focal_gamma": OPTIMAL_HYPERPARAMS["focal_gamma"],
            "use_amp": True,
            # Re-normalize weights: 0.7 + 0.2 + 0.05 = 0.95, scale to 1.0
            "loss_weights": {"focal": 0.737, "evid": 0.0, "evt": 0.211, "prec": 0.053}
        }
    },
    
    "no_evt": {
        "name": "– EVT Head", 
        "description": "Remove GPD (EVT) branch",
        "config": {
            "use_evidential": True,
            "use_evt": False,
            "use_attention_bottleneck": True,
            "use_precursor": True,
            "focal_gamma": OPTIMAL_HYPERPARAMS["focal_gamma"],
            "use_amp": True,
            # Re-normalize weights: 0.7 + 0.1 + 0.05 = 0.85, scale to 1.0
            "loss_weights": {"focal": 0.824, "evid": 0.118, "evt": 0.0, "prec": 0.059}
        }
    },
    
    "mean_pool": {
        "name": "Mean Pool (No Attention)",
        "description": "Replace attention bottleneck with mean pooling",
        "config": {
            "use_evidential": True,
            "use_evt": True,
            "use_attention_bottleneck": False,  # Use mean pooling instead
            "use_precursor": True,
            "focal_gamma": OPTIMAL_HYPERPARAMS["focal_gamma"],
            "use_amp": True,
            "loss_weights": {"focal": 0.7, "evid": 0.1, "evt": 0.2, "prec": 0.05}
        }
    },
    
    "cross_entropy": {
        "name": "Cross-Entropy (γ = 0)",
        "description": "No focal re-weighting, standard cross-entropy",
        "config": {
            "use_evidential": True,
            "use_evt": True,
            "use_attention_bottleneck": True,
            "use_precursor": True,
            "focal_gamma": 0.0,  # Disable focal loss
            "use_amp": True,
            "loss_weights": {"focal": 0.7, "evid": 0.1, "evt": 0.2, "prec": 0.05}
        }
    },
    
    "no_precursor": {
        "name": "No Precursor Head",
        "description": "Remove early-warning auxiliary head",
        "config": {
            "use_evidential": True,
            "use_evt": True,
            "use_attention_bottleneck": True,
            "use_precursor": False,
            "focal_gamma": OPTIMAL_HYPERPARAMS["focal_gamma"],
            "use_amp": True,
            # Re-normalize weights: 0.7 + 0.1 + 0.2 = 1.0 (already normalized)
            "loss_weights": {"focal": 0.7, "evid": 0.1, "evt": 0.2, "prec": 0.0}
        }
    },
    
    "fp32_training": {
        "name": "FP32 Training",
        "description": "Disable mixed precision (AMP)",
        "config": {
            "use_evidential": True,
            "use_evt": True,
            "use_attention_bottleneck": True,
            "use_precursor": True,
            "focal_gamma": OPTIMAL_HYPERPARAMS["focal_gamma"],
            "use_amp": False,  # Disable mixed precision
            "loss_weights": {"focal": 0.7, "evid": 0.1, "evt": 0.2, "prec": 0.05}
        }
    }
}

# ============================================================================
# SEQUENCE LENGTH ABLATION
# ============================================================================

SEQUENCE_LENGTH_VARIANTS = {
    "seq_5": {
        "name": "Sequence Length 5",
        "description": "5 timesteps input sequence",
        "input_shape": (5, 9)
    },
    "seq_7": {
        "name": "Sequence Length 7", 
        "description": "7 timesteps input sequence",
        "input_shape": (7, 9)
    },
    "seq_10": {
        "name": "Sequence Length 10 (Baseline)",
        "description": "10 timesteps input sequence (current)",
        "input_shape": (10, 9)
    },
    "seq_15": {
        "name": "Sequence Length 15",
        "description": "15 timesteps input sequence",
        "input_shape": (15, 9)
    },
    "seq_20": {
        "name": "Sequence Length 20",
        "description": "20 timesteps input sequence", 
        "input_shape": (20, 9)
    }
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================

EVALUATION_METRICS = [
    "TSS",           # True Skill Statistic (primary)
    "F1",            # F1 score
    "ECE",           # Expected Calibration Error (15-bin)
    "Brier",         # Brier score
    "accuracy",      # Overall accuracy
    "precision",     # Precision
    "recall",        # Recall (sensitivity)
    "specificity",   # Specificity
    "roc_auc",       # ROC AUC
    "latency_ms"     # Inference latency (milliseconds)
]

# ============================================================================
# STATISTICAL TESTING
# ============================================================================

STATISTICAL_CONFIG = {
    "bootstrap_samples": 10000,  # Number of bootstrap resamples
    "confidence_level": 0.95,    # 95% confidence intervals
    "significance_threshold": 0.05,  # p < 0.05 for significance
    "latency_samples": 1000      # Number of latency measurements
}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

OUTPUT_CONFIG = {
    "results_dir": "models/ablation/results",
    "plots_dir": "models/ablation/plots", 
    "logs_dir": "models/ablation/logs",
    "models_dir": "models/ablation/trained_models",
    "git_tag": "v4.1-ablation",
    "save_raw_csvs": True,
    "save_bootstrap_samples": True
}

# ============================================================================
# CLUSTER CONFIGURATION
# ============================================================================

CLUSTER_CONFIG = {
    "job_time_limit": "24:00:00",  # 24 hours per job
    "memory_per_job": "32GB",
    "gpu_type": "V100",
    "cpus_per_task": 8,
    "array_job_limit": 50,  # Max concurrent jobs
    "partition": "gpu"
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_variant_config(variant_name: str) -> Dict[str, Any]:
    """Get configuration for a specific ablation variant."""
    if variant_name not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown variant: {variant_name}")
    return ABLATION_VARIANTS[variant_name]["config"]

def get_sequence_config(seq_variant: str) -> Dict[str, Any]:
    """Get configuration for a specific sequence length variant."""
    if seq_variant not in SEQUENCE_LENGTH_VARIANTS:
        raise ValueError(f"Unknown sequence variant: {seq_variant}")
    return SEQUENCE_LENGTH_VARIANTS[seq_variant]

def get_all_variant_names() -> List[str]:
    """Get list of all ablation variant names."""
    return list(ABLATION_VARIANTS.keys())

def get_all_sequence_variants() -> List[str]:
    """Get list of all sequence length variant names."""
    return list(SEQUENCE_LENGTH_VARIANTS.keys())

def create_output_directories():
    """Create all necessary output directories."""
    for dir_path in OUTPUT_CONFIG.values():
        if isinstance(dir_path, str) and dir_path.endswith(('results', 'plots', 'logs', 'models')):
            os.makedirs(dir_path, exist_ok=True)

def get_experiment_name(variant: str, seed: int, sequence_length: Optional[str] = None) -> str:
    """Generate standardized experiment name."""
    if sequence_length:
        return f"ablation_{variant}_{sequence_length}_seed{seed}"
    else:
        return f"ablation_{variant}_seed{seed}"

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration consistency."""
    # Check that loss weights sum to 1.0 for each variant
    for variant_name, variant in ABLATION_VARIANTS.items():
        weights = variant["config"]["loss_weights"]
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            print(f"Warning: {variant_name} loss weights sum to {total_weight:.6f}, not 1.0")
    
    # Check that all required directories exist
    create_output_directories()
    
    print("✅ Ablation configuration validated successfully")

if __name__ == "__main__":
    validate_config()
    print(f"📊 Configured {len(ABLATION_VARIANTS)} ablation variants")
    print(f"📏 Configured {len(SEQUENCE_LENGTH_VARIANTS)} sequence length variants") 
    print(f"🎲 Using {len(RANDOM_SEEDS)} random seeds")
    print(f"🎯 Primary target: {PRIMARY_TARGET['flare_class']}-class, {PRIMARY_TARGET['time_window']}h") 