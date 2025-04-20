"""Simple test to verify that all modules can be imported.

This is a basic sanity check for CI.
"""


def test_solar_knowledge_imports():
    """Test that all solar_knowledge modules can be imported."""
    import solar_knowledge
    import solar_knowledge.data
    import solar_knowledge.eval_full
    import solar_knowledge.smoke_train

    # Verify __version__ exists
    assert hasattr(solar_knowledge, "__version__")
    assert isinstance(solar_knowledge.__version__, str)


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
