import tensorflow as tf
from tensorflow.keras import layers

def nig_head(x, name=None):
    """Return μ, ν, α, β – all positive where needed."""
    # Ensure x has sufficient feature dimensions
    x_shape = tf.shape(x)
    batch_dim = x_shape[0]
    
    # Project to 4D space (mu, v, alpha, beta)
    # Using a single Dense layer with 4 output units, rather than 4 separate layers
    params = layers.Dense(4, name="evidential_params")(x)
    
    # Split into separate components for transformations
    mu, logv, logalpha, logbeta = tf.split(params, 4, axis=-1)
    
    # Apply appropriate activations
    v = tf.nn.softplus(logv)             # ν > 0
    alpha = 1 + tf.nn.softplus(logalpha) # α > 1
    beta = tf.nn.softplus(logbeta)       # β > 0
    
    # Concatenate back together
    out = tf.concat([mu, v, alpha, beta], axis=-1)
    
    # The final shape should be (batch_size, 4)
    # Ensure the shape is explicitly set
    out = tf.reshape(out, [-1, 4])
    
    # Add a name for debugging purposes
    return layers.Activation('linear', name=name)(out)

def evidential_nll(y_true, evid):
    """Negative log‑likelihood for binary‐classification evidential head.
    
    Takes a tensor of shape (batch_size, 4) containing [μ, ν, α, β]
    where activations have already been applied.
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
    
    # Ensure parameters are within valid ranges
    # Apply clipping for numerical stability
    α = tf.clip_by_value(α, 1.0 + 1e-6, 1e6)
    β = tf.clip_by_value(β, 1e-6, 1e6)
    v = tf.clip_by_value(v, 1e-6, 1e6)
    
    # Calculate predictive variance with clipping
    S = β*(1+v)/(α)
    
    # Use Kumaraswamy approx to Beta NLL for numerical stability
    eps = tf.keras.backend.epsilon()
    nll = - y_true   * tf.math.log(p + eps) \
          - (1-y_true)*tf.math.log(1-p + eps) \
          + 0.5*tf.math.log(S + eps)
    
    return tf.reduce_mean(nll) 