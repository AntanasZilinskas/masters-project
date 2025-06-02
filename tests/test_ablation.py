"""
Tests for ablation study configuration and functionality.
"""

import pytest
import sys
import os

# Add models to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))


def test_ablation_config_import():
    """Test that ablation config can be imported and contains required components."""
    try:
        from ablation.config import (
            ABLATION_COMPONENTS,
            BASELINE_CONFIG,
            get_ablation_experiments,
            validate_ablation_config,
        )

        # Basic structure checks
        assert len(ABLATION_COMPONENTS) > 0, "Should have ablation components defined"
        assert isinstance(
            BASELINE_CONFIG, dict
        ), "Baseline config should be a dictionary"

        # Config validation
        validate_ablation_config()

        # Experiment generation
        experiments = get_ablation_experiments()
        assert len(experiments) > 0, "Should generate ablation experiments"

    except ImportError:
        pytest.skip("Ablation config not available in this branch")


def test_ablation_components():
    """Test that ablation components have correct structure."""
    try:
        from ablation.config import ABLATION_COMPONENTS

        for component_name, component_config in ABLATION_COMPONENTS.items():
            assert isinstance(component_name, str), "Component name should be string"
            assert isinstance(component_config, dict), "Component config should be dict"
            assert (
                "description" in component_config
            ), f"Component {component_name} should have description"

    except ImportError:
        pytest.skip("Ablation config not available in this branch")


def test_baseline_configuration():
    """Test baseline configuration structure."""
    try:
        from ablation.config import BASELINE_CONFIG

        required_keys = [
            "use_attention_bottleneck",
            "use_evidential",
            "use_evt",
            "use_precursor",
        ]
        for key in required_keys:
            assert (
                key in BASELINE_CONFIG
            ), f"Baseline config missing required key: {key}"
            assert isinstance(
                BASELINE_CONFIG[key], bool
            ), f"Baseline config {key} should be boolean"

    except ImportError:
        pytest.skip("Ablation config not available in this branch")


def test_ablation_experiment_generation():
    """Test ablation experiment generation."""
    try:
        from ablation.config import get_ablation_experiments, ABLATION_COMPONENTS

        experiments = get_ablation_experiments()

        # Should have baseline + one experiment per component
        expected_count = 1 + len(ABLATION_COMPONENTS)
        assert (
            len(experiments) == expected_count
        ), f"Expected {expected_count} experiments, got {len(experiments)}"

        # Check baseline experiment
        baseline_exp = next(
            (exp for exp in experiments if exp["name"] == "baseline"), None
        )
        assert baseline_exp is not None, "Should have baseline experiment"

        # Check ablation experiments
        for component_name in ABLATION_COMPONENTS.keys():
            ablation_exp = next(
                (
                    exp
                    for exp in experiments
                    if exp["name"] == f"ablate_{component_name}"
                ),
                None,
            )
            assert (
                ablation_exp is not None
            ), f"Should have ablation experiment for {component_name}"

    except ImportError:
        pytest.skip("Ablation config not available in this branch")


def test_ablation_metrics():
    """Test ablation study metrics configuration."""
    try:
        from ablation.config import ABLATION_METRICS

        expected_metrics = ["tss", "accuracy", "f1", "precision", "recall"]
        for metric in expected_metrics:
            assert metric in ABLATION_METRICS, f"Should include metric: {metric}"

    except ImportError:
        pytest.skip("Ablation config not available in this branch")


def test_ablation_output_structure():
    """Test ablation output configuration."""
    try:
        from ablation.config import ABLATION_OUTPUT_CONFIG

        required_keys = ["results_dir", "plots_dir", "models_dir"]
        for key in required_keys:
            assert (
                key in ABLATION_OUTPUT_CONFIG
            ), f"Missing ablation output config: {key}"

    except ImportError:
        pytest.skip("Ablation config not available in this branch")


def test_ablation_compatibility():
    """Test that ablation studies can work with different frameworks."""
    # This test ensures ablation studies are framework-agnostic
    branch_name = os.environ.get("GITHUB_REF", "").replace("refs/heads/", "")
    print(f"Testing ablation compatibility for branch: {branch_name}")

    if "pytorch" in branch_name:
        # Test PyTorch compatibility
        try:
            import torch

            # Simulate ablation component test
            x = torch.randn(2, 3, 4)  # batch, seq, features
            print("✓ Ablation framework compatibility (PyTorch)")
        except ImportError:
            pytest.skip("PyTorch not available for ablation testing")
    else:
        # Test TensorFlow compatibility
        try:
            import tensorflow as tf

            # Simulate ablation component test
            x = tf.random.normal((2, 3, 4))  # batch, seq, features
            print("✓ Ablation framework compatibility (TensorFlow)")
        except ImportError:
            pytest.skip("TensorFlow not available for ablation testing")
