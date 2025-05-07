import tensorflow as tf
from tensorflow.keras import layers

def nig_head(x, name=None):
    """
    Normal-Inverse-Gamma (NIG) head for evidential uncertainty.
    
    Returns μ, ν, α, β – all positive where needed.
    
    Args:
        x: Input features tensor
        name: Optional name for output tensor
        
    Returns:
        Tensor with shape [batch_size, 4] containing NIG parameters
    """
    # Ensure x has sufficient feature dimensions
    x_shape = tf.shape(x)
    batch_dim = x_shape[0]
    
    # Add a hidden layer for better parameter estimation
    hidden = tf.keras.layers.Dense(
        32,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name=f'{name}_hidden' if name else None
    )(x)
    
    # Apply batch normalization and dropout for more stable training
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Dropout(0.2)(hidden)
    
    # Project to 4D space (mu, v, alpha, beta)
    # Using a single Dense layer with 4 output units
    params = layers.Dense(4, name="evidential_params")(hidden)
    
    # Split into separate components for transformations
    mu, logv, logalpha, logbeta = tf.split(params, 4, axis=-1)
    
    # Apply appropriate activations with improved stability
    # Add small offsets to avoid numerical issues
    v = tf.nn.softplus(logv) + 0.1             # ν > 0.1
    alpha = 1.0 + tf.nn.softplus(logalpha)     # α > 1.0
    beta = tf.nn.softplus(logbeta) + 0.1       # β > 0.1
    
    # Concatenate back together
    out = tf.concat([mu, v, alpha, beta], axis=-1)
    
    # The final shape should be (batch_size, 4)
    # Ensure the shape is explicitly set
    out = tf.reshape(out, [-1, 4])
    
    # Add a name for debugging purposes
    return layers.Activation('linear', name=name)(out)

def evidential_nll(y_true, evid, kl_weight=0.01):
    """
    Negative log‑likelihood for binary‐classification evidential head with KL regularization.
    
    Takes a tensor of shape (batch_size, 4) containing [μ, ν, α, β]
    where activations have already been applied.
    
    Args:
        y_true: Binary ground truth (0/1)
        evid: Tensor of shape [batch_size, 4] containing NIG parameters
        kl_weight: Weight for KL divergence regularization
        
    Returns:
        NLL loss with KL regularization
    """
    # Handle potential shape issues
    # Ensure evid has shape (batch_size, 4)
    evid_shape = tf.shape(evid)
    
    # Reshape y_true to ensure it's always (batch_size, 1)
    y_true = tf.reshape(y_true, [-1, 1])
    
    # If the last dimension isn't 4, something is wrong
    tf.debugging.assert_equal(evid_shape[-1], 4, 
                             message="Evidential parameters must have 4 components")
    
    # Split into NIG parameters (already activated)
    mu, v, α, β = tf.split(evid, 4, axis=-1)  # Each has shape (batch_size, 1)
    
    # Convert mu to probability
    p = tf.nn.sigmoid(mu)
    
    # Ensure parameters are within valid ranges with improved clipping
    # Apply clipping for numerical stability
    α = tf.clip_by_value(α, 1.0 + 1e-6, 1e6)
    β = tf.clip_by_value(β, 1e-6, 1e6)
    v = tf.clip_by_value(v, 1e-6, 1e6)
    
    # Calculate predictive variance
    # S = β*(1+v)/(α)
    S = β*(1+v)/(α - 1.0 + 1e-6)  # Corrected variance calculation with safeguard
    
    # Small epsilon for log stability
    eps = 1e-7
    
    # Calculate NLL components with better numerical stability
    # Binary cross-entropy term
    bce_term = - y_true * tf.math.log(p + eps) - (1-y_true)*tf.math.log(1-p + eps)
    
    # Variance term
    var_term = 0.5*tf.math.log(S + eps)
    
    # Add a small regularization term to ensure positive values
    # Try to encourage lower variance for correct predictions
    # This penalizes high uncertainty when the model is correct
    # Cast boolean to float for multiplication
    is_correct_prediction = tf.cast(tf.abs(y_true - p) < 0.2, tf.float32)
    var_reg = 0.1 * S * is_correct_prediction
    
    # Combine NLL terms
    nll = tf.abs(bce_term) + tf.abs(var_term) + var_reg
    
    # Add KL regularization to prevent "infinite evidence" collapse
    # This comes from the prior in the evidential framework
    # KL divergence between NIG and prior NIG with fixed parameters
    kl_term = kl_divergence_nig(α, β, v, mu, 1.0, 0.1, 0.1, 0.0)
    
    # Final loss with regularization
    return tf.reduce_mean(nll) + kl_weight * tf.reduce_mean(kl_term)

def kl_divergence_nig(α1, β1, v1, μ1, α2, β2, v2, μ2):
    """
    KL divergence between two Normal-Inverse-Gamma distributions.
    
    This penalizes overconfident predictions by adding a regularization term
    that keeps the parameters close to a prior distribution.
    
    Args:
        α1, β1, v1, μ1: Parameters of the first NIG distribution
        α2, β2, v2, μ2: Parameters of the second NIG distribution (prior)
        
    Returns:
        KL divergence term
    """
    # Small epsilon for numerical stability
    eps = 1e-7
    
    # Term 1: Log ratio of β values and difference of α's
    term1 = α2 * tf.math.log(β1 / (β2 + eps) + eps) + tf.math.lgamma(α1) - tf.math.lgamma(α2)
    
    # Term 2: Digamma term
    term2 = (α1 - α2) * tf.math.digamma(α1)
    
    # Term 3: Ratio term with v values
    term3 = 0.5 * tf.math.log(v2 / (v1 + eps) + eps)
    
    # Term 4: Scaled variance term
    term4 = 0.5 * v2 * (μ1 - μ2)**2 / (v1 + eps)
    
    # Term 5: Ratio of α and β
    term5 = α1 * (β2 - β1) / (β1 + eps)
    
    # Combine all terms
    kl = term1 + term2 + term3 + term4 + term5
    
    # Ensure KL is non-negative and finite
    kl = tf.maximum(kl, 0.0)
    kl = tf.minimum(kl, 1e6)  # Cap very large values
    
    return kl 