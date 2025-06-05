"""
Tests for hyperparameter optimization (HPO) configuration and functionality.
"""

import pytest
import sys
import os

# Add models to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))


def test_hpo_config_import():
    """Test that HPO config can be imported and contains required components."""
    try:
        from hpo.config import (
            HPO_SEARCH_SPACE,
            HPO_OBJECTIVE_CONFIG,
            validate_hpo_config,
        )

        # Basic structure checks
        assert isinstance(
            HPO_SEARCH_SPACE, dict
        ), "HPO search space should be a dictionary"
        assert len(HPO_SEARCH_SPACE) > 0, "Should have HPO search space defined"
        assert isinstance(
            HPO_OBJECTIVE_CONFIG, dict
        ), "HPO objective config should be a dictionary"

        # Config validation
        validate_hpo_config()

    except ImportError:
        pytest.skip("HPO config not available in this branch")


def test_hpo_search_space_structure():
    """Test that HPO search space has correct structure."""
    try:
        from hpo.config import HPO_SEARCH_SPACE

        # Check for key hyperparameters
        expected_params = ["learning_rate", "batch_size", "dropout", "embed_dim"]
        for param in expected_params:
            assert param in HPO_SEARCH_SPACE, f"Missing hyperparameter: {param}"

            param_config = HPO_SEARCH_SPACE[param]
            assert isinstance(
                param_config, (dict, list, tuple)
            ), f"Parameter {param} config should be dict, list, or tuple"

            if isinstance(param_config, dict):
                assert "type" in param_config, f"Parameter {param} should have type"

                param_type = param_config["type"]
                if param_type == "float":
                    assert (
                        "low" in param_config and "high" in param_config
                    ), f"Float param {param} needs low/high"
                elif param_type == "int":
                    assert (
                        "low" in param_config and "high" in param_config
                    ), f"Int param {param} needs low/high"
                elif param_type == "categorical":
                    assert (
                        "choices" in param_config
                    ), f"Categorical param {param} needs choices"

    except ImportError:
        pytest.skip("HPO config not available in this branch")


def test_hpo_objective_configuration():
    """Test HPO objective configuration."""
    try:
        from hpo.config import HPO_OBJECTIVE_CONFIG

        required_keys = ["metric", "direction", "n_trials", "timeout"]
        for key in required_keys:
            assert key in HPO_OBJECTIVE_CONFIG, f"Missing HPO objective config: {key}"

        # Validate specific values
        assert HPO_OBJECTIVE_CONFIG["metric"] in [
            "tss",
            "f1",
            "accuracy",
        ], "Invalid optimization metric"
        assert HPO_OBJECTIVE_CONFIG["direction"] in [
            "maximize",
            "minimize",
        ], "Invalid optimization direction"
        assert isinstance(
            HPO_OBJECTIVE_CONFIG["n_trials"], int
        ), "n_trials should be integer"
        assert HPO_OBJECTIVE_CONFIG["n_trials"] > 0, "n_trials should be positive"

    except ImportError:
        pytest.skip("HPO config not available in this branch")


def test_hpo_study_configuration():
    """Test HPO study configuration."""
    try:
        from hpo.config import HPO_STUDY_CONFIG

        required_keys = ["storage", "study_name", "sampler", "pruner"]
        for key in required_keys:
            assert key in HPO_STUDY_CONFIG, f"Missing HPO study config: {key}"

    except ImportError:
        pytest.skip("HPO config not available in this branch")


def test_optuna_integration():
    """Test that Optuna can be imported and basic functionality works."""
    try:
        import optuna

        # Test basic study creation
        study = optuna.create_study(direction="maximize")
        assert study is not None, "Should create optuna study"

        # Test simple objective function
        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return (x - 2) ** 2

        study.optimize(objective, n_trials=3)
        assert len(study.trials) == 3, "Should complete 3 trials"

    except ImportError:
        pytest.skip("Optuna not available for testing")


def test_hpo_parameter_validation():
    """Test HPO parameter validation function."""
    try:
        from hpo.config import validate_hpo_parameters

        # Test valid parameters
        valid_params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "dropout": 0.2,
            "embed_dim": 128,
        }

        result = validate_hpo_parameters(valid_params)
        assert result == True, "Valid parameters should pass validation"

        # Test invalid parameters
        invalid_params = {
            "learning_rate": -0.001,  # negative learning rate
            "batch_size": 0,  # zero batch size
            "dropout": 1.5,  # dropout > 1
        }

        result = validate_hpo_parameters(invalid_params)
        assert result == False, "Invalid parameters should fail validation"

    except ImportError:
        pytest.skip("HPO config not available in this branch")


def test_hpo_output_configuration():
    """Test HPO output configuration."""
    try:
        from hpo.config import HPO_OUTPUT_CONFIG

        required_keys = ["results_dir", "best_models_dir", "studies_dir", "plots_dir"]
        for key in required_keys:
            assert key in HPO_OUTPUT_CONFIG, f"Missing HPO output config: {key}"

    except ImportError:
        pytest.skip("HPO config not available in this branch")


def test_hpo_framework_compatibility():
    """Test that HPO works with different ML frameworks."""
    branch_name = os.environ.get("GITHUB_REF", "").replace("refs/heads/", "")
    print(f"Testing HPO framework compatibility for branch: {branch_name}")

    # Test that basic optimization concepts work regardless of framework
    try:
        import optuna

        if "pytorch" in branch_name:
            # Test PyTorch-style HPO
            try:
                import torch

                def pytorch_objective(trial):
                    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
                    model = torch.nn.Linear(10, 1)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    return lr  # Mock objective value

                study = optuna.create_study()
                study.optimize(pytorch_objective, n_trials=2)
                print("✓ HPO PyTorch compatibility verified")

            except ImportError:
                pytest.skip("PyTorch not available for HPO testing")
        else:
            # Test TensorFlow-style HPO
            try:
                import tensorflow as tf

                def tensorflow_objective(trial):
                    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
                    model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
                    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
                    return lr  # Mock objective value

                study = optuna.create_study()
                study.optimize(tensorflow_objective, n_trials=2)
                print("✓ HPO TensorFlow compatibility verified")

            except ImportError:
                pytest.skip("TensorFlow not available for HPO testing")

    except ImportError:
        pytest.skip("Optuna not available for framework compatibility testing")
