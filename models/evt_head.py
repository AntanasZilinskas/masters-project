"""
Extreme Value Theory (EVT) Module for EVEREST model.

This module provides tools for Generalized Pareto Distribution (GPD) modeling
of the upper tail of rare events (solar flares). This implementation includes
improved numerical stability and proper threshold-based handling.
"""

import tensorflow as tf

def gpd_head(x, name=None):
    """
    Create a GPD parameter estimation head with improved constraints.
    
    Args:
        x: Input tensor with shape [batch_size, features]
        name: Optional name for the output tensor
        
    Returns:
        Tensor with shape [batch_size, 2] containing [ξ, σ] for each sample
    """
    # Use a more stable MLP architecture
    # Add a hidden layer with stronger regularization for better generalization
    hidden = tf.keras.layers.Dense(
        32,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name=f'{name}_hidden' if name else None
    )(x)
    
    # Add dropout for better regularization
    hidden = tf.keras.layers.Dropout(0.3)(hidden)
    
    # Add batch normalization for more stable training
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    
    # Project to 2D space (shape, scale)
    dense = tf.keras.layers.Dense(
        2,
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
        name=f'{name}_dense' if name else None
    )(hidden)
    
    # Use tanh for shape parameter to constrain between -0.9 and 0.9
    # This improves numerical stability while allowing flexibility
    shape = tf.keras.layers.Lambda(
        lambda x: tf.tanh(x[..., 0:1]) * 0.9,
        name=f'{name}_shape' if name else None
    )(dense)
    
    # Use softplus with offset for scale parameter to ensure positivity
    # Adding 0.1 prevents scale from getting too close to zero
    scale = tf.keras.layers.Lambda(
        lambda x: tf.nn.softplus(x[..., 1:2]) + 0.1,
        name=f'{name}_scale' if name else None
    )(dense)
    
    # Combine parameters
    params = tf.keras.layers.Concatenate(axis=-1, name=name)([shape, scale])
    
    return params

def evt_loss(logits, evt_params, threshold=0.5, mask_quantile=0.95):
    """
    Extreme Value Theory (EVT) loss function for tail modeling with masking.
    
    Args:
        logits: Logits from model prediction, shape [batch_size, 1]
        evt_params: GPD parameters [ξ, σ], shape [batch_size, 2]
        threshold: Base threshold for tail modeling (default 0.5)
        mask_quantile: Quantile for automatic threshold adjustment (default 0.95)
            Higher values (e.g., 0.98) focus more on extreme values
            
    Returns:
        EVT loss value (scalar)
    """
    # Ensure inputs have the right shape
    logits = tf.reshape(logits, [-1, 1])
    evt_params = tf.reshape(evt_params, [-1, 2])
    
    # Unpack GPD parameters
    shape = evt_params[:, 0:1]  # ξ (shape)
    scale = evt_params[:, 1:2]  # σ (scale)
    
    # If mask_quantile is provided, use it to determine dynamic threshold
    # This ensures we're always modeling the right tail (top X% of samples)
    if mask_quantile > 0:
        dyn_threshold = tf.maximum(
            threshold,
            tfp_percentile(logits, mask_quantile * 100)
        )
    else:
        dyn_threshold = threshold
    
    # Create mask for tail values (those exceeding threshold)
    tail_mask = tf.cast(logits > dyn_threshold, tf.float32)
    
    # Count how many samples exceed threshold
    n_exceedances = tf.reduce_sum(tail_mask)
    
    # Apply threshold to identify exceedances
    exceedance = tf.maximum(logits - dyn_threshold, 0.0)
    
    # Small constant for numerical stability
    eps = 1e-6
    
    # For ξ ≈ 0, use exponential approximation
    # For |ξ| > eps, use standard GPD formula with safe division
    safe_shape = tf.where(
        tf.abs(shape) < eps,
        tf.ones_like(shape) * eps * tf.sign(shape + eps),
        shape
    )
    
    # Calculate negative log-likelihood for GPD with better numerical stability
    # When exceedance > 0, calculate NLL; otherwise, use 0
    log_scale = tf.math.log(scale + eps)
    
    # Term 1: log(scale)
    term1 = log_scale
    
    # Term 2: (1 + 1/shape) * log(1 + shape*exceedance/scale)
    # Handle the case where shape is close to zero separately
    term2_denom = 1.0 + safe_shape * exceedance / (scale + eps)
    term2 = (1.0 + 1.0 / safe_shape) * tf.math.log(tf.maximum(term2_denom, eps))
    
    # Calculate NLL only for exceedances
    nll = tf.where(
        exceedance > 0,
        term1 + term2,
        tf.zeros_like(exceedance)
    ) * tail_mask
    
    # Add regularization to encourage sensible parameter values
    # Penalize large scale and extreme shape values
    reg_scale = 0.1 * tf.reduce_mean(scale)
    reg_shape = 0.2 * tf.reduce_mean(tf.abs(shape))
    
    # If we have any exceedances, calculate proper loss
    # Otherwise, just use regularization to avoid NaNs
    if n_exceedances > 0:
        evt_loss_val = tf.reduce_sum(nll) / (n_exceedances + eps) + reg_scale + reg_shape
    else:
        # If no exceedances, focus on regularization only
        evt_loss_val = reg_scale + reg_shape + 0.01  # Small constant to avoid zero loss
    
    return evt_loss_val

def tfp_percentile(x, q):
    """TensorFlow implementation of percentile function."""
    # Sort the tensor
    x_sorted = tf.sort(x, axis=0)
    
    # Get the index
    frac = q / 100.0
    N = tf.cast(tf.shape(x)[0], tf.float32)
    index = tf.floor(frac * (N - 1))
    index = tf.cast(index, tf.int32)
    
    # Get the value at index
    return x_sorted[index][0]

def gpd_threshold_probability(logits, evt_params, threshold, return_exceedances=False):
    """
    Calculate the probability of exceeding a threshold using GPD parameters.
    
    Args:
        logits: Logits from model prediction
        evt_params: GPD parameters [ξ, σ]
        threshold: Threshold to calculate exceedance probability
        return_exceedances: Whether to return the exceedances as well
        
    Returns:
        Probability of exceeding the threshold
    """
    # Ensure inputs have the right shape
    logits = tf.reshape(logits, [-1, 1])
    evt_params = tf.reshape(evt_params, [-1, 2])
    
    # Unpack GPD parameters
    shape = evt_params[:, 0:1]  # ξ (shape)
    scale = evt_params[:, 1:2]  # σ (scale)
    
    # Calculate exceedances
    exceedance = tf.maximum(logits - threshold, 0.0)
    
    # Small constant for numerical stability
    eps = 1e-6
    
    # Safe shape parameter
    safe_shape = tf.where(
        tf.abs(shape) < eps,
        tf.ones_like(shape) * eps * tf.sign(shape + eps),
        shape
    )
    
    # Calculate GPD survival function (probability of exceeding x)
    # P(X > x | X > u) = (1 + ξx/σ)^(-1/ξ) for ξ ≠ 0
    # P(X > x | X > u) = exp(-x/σ) for ξ = 0
    
    # For ξ ≈ 0, use exponential approximation
    exponential_prob = tf.exp(-exceedance / (scale + eps))
    
    # For other ξ values, use standard GPD formula
    term = 1.0 + safe_shape * exceedance / (scale + eps)
    gpd_prob = tf.pow(tf.maximum(term, eps), -1.0 / safe_shape)
    
    # Combine based on shape parameter
    survival_prob = tf.where(
        tf.abs(shape) < eps,
        exponential_prob,
        gpd_prob
    )
    
    # Apply mask for valid exceedances
    valid_mask = tf.cast(exceedance > 0, tf.float32)
    masked_prob = survival_prob * valid_mask
    
    if return_exceedances:
        return masked_prob, exceedance
    else:
        return masked_prob 