"""
Tests for training configuration and core training functionality.
"""

import pytest
import sys
import os

# Add models to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))


def test_training_config_import():
    """Test that training config can be imported and contains required components."""
    try:
        from training.config import (
            TRAINING_TARGETS,
            RANDOM_SEEDS,
            FIXED_ARCHITECTURE,
            TRAINING_HYPERPARAMS,
            get_all_experiments,
            validate_config
        )
        
        # Basic structure checks
        assert len(TRAINING_TARGETS) > 0, "Should have training targets defined"
        assert len(RANDOM_SEEDS) > 0, "Should have random seeds defined"
        assert "input_shape" in FIXED_ARCHITECTURE, "Should have input_shape in architecture"
        assert "epochs" in TRAINING_HYPERPARAMS, "Should have epochs in hyperparams"
        
        # Config validation
        validate_config()
        
        # Experiment generation
        experiments = get_all_experiments()
        expected_count = len(TRAINING_TARGETS) * len(RANDOM_SEEDS)
        assert len(experiments) == expected_count, f"Expected {expected_count} experiments, got {len(experiments)}"
        
    except ImportError:
        pytest.skip("Training config not available in this branch")


def test_training_target_structure():
    """Test that training targets have correct structure."""
    try:
        from training.config import TRAINING_TARGETS
        
        for target in TRAINING_TARGETS:
            assert "flare_class" in target, "Training target should have flare_class"
            assert "time_window" in target, "Training target should have time_window"
            assert target["flare_class"] in ["C", "M", "M5"], f"Invalid flare class: {target['flare_class']}"
            assert target["time_window"] in ["24", "48", "72"], f"Invalid time window: {target['time_window']}"
            
    except ImportError:
        pytest.skip("Training config not available in this branch")


def test_experiment_name_generation():
    """Test experiment name generation."""
    try:
        from training.config import get_experiment_name
        
        name = get_experiment_name("M5", "72", 0)
        assert name == "everest_M5_72h_seed0", f"Unexpected experiment name: {name}"
        
    except ImportError:
        pytest.skip("Training config not available in this branch")


def test_threshold_configuration():
    """Test threshold optimization configuration."""
    try:
        from training.config import THRESHOLD_CONFIG, get_threshold_search_points
        
        assert 0.0 < THRESHOLD_CONFIG["search_range"][0] < THRESHOLD_CONFIG["search_range"][1] < 1.0
        assert THRESHOLD_CONFIG["search_points"] > 10, "Should have reasonable number of search points"
        
        thresholds = get_threshold_search_points()
        assert len(thresholds) == THRESHOLD_CONFIG["search_points"]
        assert all(0.0 <= t <= 1.0 for t in thresholds), "All thresholds should be in [0,1]"
        
    except ImportError:
        pytest.skip("Training config not available in this branch")


def test_balanced_score_calculation():
    """Test balanced score calculation for threshold optimization."""
    try:
        from training.config import calculate_balanced_score, BALANCED_WEIGHTS
        
        # Test with perfect metrics
        perfect_metrics = {metric: 1.0 for metric in BALANCED_WEIGHTS.keys()}
        score = calculate_balanced_score(perfect_metrics)
        expected_score = sum(BALANCED_WEIGHTS.values())
        assert abs(score - expected_score) < 1e-6, f"Expected score {expected_score}, got {score}"
        
        # Test with zero metrics
        zero_metrics = {metric: 0.0 for metric in BALANCED_WEIGHTS.keys()}
        score = calculate_balanced_score(zero_metrics)
        assert abs(score) < 1e-6, f"Expected score ~0, got {score}"
        
    except ImportError:
        pytest.skip("Training config not available in this branch")


def test_output_directories():
    """Test output directory configuration."""
    try:
        from training.config import OUTPUT_CONFIG, create_output_directories
        
        # Check required keys
        required_keys = ["base_dir", "results_dir", "models_dir", "logs_dir"]
        for key in required_keys:
            assert key in OUTPUT_CONFIG, f"Missing required output config: {key}"
        
        # Test directory creation (this should not fail)
        try:
            create_output_directories()
        except Exception as e:
            pytest.fail(f"Directory creation failed: {e}")
            
    except ImportError:
        pytest.skip("Training config not available in this branch")


def test_branch_compatibility():
    """Test that we can detect the current branch and framework."""
    # Test framework availability based on branch
    branch_name = os.environ.get('GITHUB_REF', '').replace('refs/heads/', '')
    
    if 'pytorch' in branch_name:
        # Should have PyTorch available
        try:
            import torch
            assert torch.__version__ is not None, "PyTorch should be available and have version"
            print(f"✓ PyTorch {torch.__version__} detected for branch: {branch_name}")
        except ImportError:
            pytest.fail("PyTorch should be available in pytorch-rewrite branch")
    else:
        # Should have TensorFlow available  
        try:
            import tensorflow as tf
            assert tf.__version__ is not None, "TensorFlow should be available and have version"
            print(f"✓ TensorFlow {tf.__version__} detected for branch: {branch_name}")
        except ImportError:
            pytest.fail("TensorFlow should be available in main branch") 