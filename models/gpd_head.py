import tensorflow as tf
from tensorflow.keras import layers

def gpd_head(x, name=None):
    """Outputs ξ (shape) unconstrained, σ (scale>0)"""
    # For testing purposes only - simplify to just output 2 parameters
    # with proper shapes
    params = layers.Dense(2, name=name + "_params")(x)
    return params 