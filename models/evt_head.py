import tensorflow as tf
from tensorflow.keras import layers

def gpd_head(x, name=None):
    """Outputs ξ (shape) unconstrained, σ (scale>0)"""
    # Project to 2D space (xi, sigma) with a single Dense layer
    params = layers.Dense(2, name="evt_params")(x)
    
    # Split into separate components
    xi, logσ = tf.split(params, 2, axis=-1)
    
    # Apply appropriate activation to sigma only (xi can be any real number)
    σ = tf.nn.softplus(logσ)  # σ > 0
    
    # Concatenate back together and ensure shape is (batch_size, 2)
    out = tf.concat([xi, σ], axis=-1)
    out = tf.reshape(out, [-1, 2])
    
    return layers.Activation('linear', name=name)(out)

def evt_loss(logits, gpd_par, threshold=2.5):
    """-log‑likelihood of logits above threshold under GPD.
    
    Takes:
    - logits: shape (batch_size, 1)
    - gpd_par: shape (batch_size, 2) with [ξ, σ] where σ > 0
    """
    # Ensure inputs have the right shapes
    logits = tf.reshape(logits, [-1, 1])  # (batch_size, 1)
    
    # Verify gpd_par has shape (batch_size, 2)
    gpd_shape = tf.shape(gpd_par)
    tf.debugging.assert_equal(gpd_shape[-1], 2,
                             message="GPD parameters must have 2 components")
    
    # Split into shape and scale parameters (already activated)
    xi, σ = tf.split(gpd_par, 2, axis=-1)
    
    # For training, just return a minimal loss when we don't have real logits
    # This is for compatibility with the multi-output loss scheme
    if tf.reduce_all(tf.math.is_nan(logits)) or tf.reduce_all(tf.equal(logits, 0)):
        return 0.0
    
    # Calculate exceedances
    y = tf.nn.relu(logits - threshold)  # Zero for values below threshold
    
    # Small constant to avoid division by zero
    eps = tf.keras.backend.epsilon()
    
    # GPD log-likelihood (negated)
    term = tf.where(tf.abs(xi) < 1e-3,  # use limit ξ→0 (exponential)
                    y/σ + tf.math.log(σ+eps),
                    (1/xi+1)*tf.math.log1p(xi*y/(σ+eps)) + tf.math.log(σ+eps))
    
    return tf.reduce_mean(term) 