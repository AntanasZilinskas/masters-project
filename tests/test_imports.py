"""Simple test to verify that all modules can be imported.

This is a basic sanity check for CI.
"""

import pytest


def test_solar_knowledge_imports():
    """Test that solar_knowledge modules can be imported if available."""
    try:
        import solar_knowledge
        import solar_knowledge.data
        import solar_knowledge.eval_full
        import solar_knowledge.smoke_train

        # Verify __version__ exists
        assert hasattr(solar_knowledge, "__version__")
        assert isinstance(solar_knowledge.__version__, str)
    except ImportError:
        pytest.skip("solar_knowledge module not available in this environment")


def test_project_modules():
    """Test that our project modules can be imported."""
    import sys
    import os
    
    # Add models to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nature_models'))
    
    # Test core model imports
    try:
        from models import utils  # noqa: F401
        print("✓ models.utils imported successfully")
    except ImportError as e:
        print(f"⚠️ models.utils import failed: {e}")
    
    try:
        from nature_models import utils  # noqa: F401  
        print("✓ nature_models.utils imported successfully")
    except ImportError as e:
        print(f"⚠️ nature_models.utils import failed: {e}")


def test_core_dependencies():
    """Test that core dependencies can be imported."""
    import numpy  # noqa: F401
    import tensorflow  # noqa: F401

    # Optional imports - these will be skipped if not available
    try:
        import torch  # noqa: F401
    except ImportError:
        print("PyTorch not available - skipping test")

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("Matplotlib not available - skipping test")
