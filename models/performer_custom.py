"""
Performer implementation with FAVOR+ linear attention mechanism
Based on "Rethinking Attention with Performers" (https://arxiv.org/abs/2009.14794)

This module provides a drop-in replacement for standard attention with O(L) complexity 
instead of O(L²), allowing for much longer sequence processing.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import math

def orthogonal_random_matrix(n_rows, n_cols, seed=None):
    """
    Creates a random orthogonal matrix for random projections.
    """
    random_state = np.random.RandomState(seed)
    H = np.random.normal(0.0, 1.0, (n_rows, n_cols))
    Q, R = np.linalg.qr(H)
    return Q

class RandomFourierFeatures(tf.keras.layers.Layer):
    """
    Approximates exp(-||x-y||^2/(2*sigma^2)) kernel using random features.
    """
    def __init__(self, output_dim, scale=1.0, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.scale = scale
        self.seed = seed

    def build(self, input_shape):
        input_dim = input_shape[-1]
        kernel_shape = (input_dim, self.output_dim)
    
        # Create orthogonal random features
        kernel_initializer = tf.keras.initializers.Orthogonal(gain=1.0, seed=self.seed)
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=kernel_initializer,
            trainable=False
        )
    
        # Scaling factor
        self.bias = self.add_weight(
            name='bias',
            shape=(self.output_dim,),
            initializer=tf.keras.initializers.RandomUniform(
                minval=0, maxval=2 * np.pi, seed=self.seed),
            trainable=False
        )
        
        self.scale_factor = tf.sqrt(2.0 / tf.cast(self.output_dim, tf.float32))
        
    def call(self, inputs):
        # Project inputs to random feature space
        projection = tf.matmul(inputs * self.scale, self.kernel)
        projection = projection + self.bias
        
        # Apply non-linearity
        return tf.math.cos(projection) * self.scale_factor

class FastSelfAttention(tf.keras.layers.Layer):
    """
    FAVOR+ self-attention mechanism with O(L) complexity.
    """
    def __init__(self, num_heads, key_dim, feature_dim=256, kernel_scale=1.0, 
                 dropout=0.0, use_relu=False, causal=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.feature_dim = feature_dim
        self.kernel_scale = kernel_scale
        self.dropout_rate = dropout
        self.use_relu = use_relu
        self.causal = causal
        
        # Total dimension for all heads
        self.total_key_dim = num_heads * key_dim
        
        # Normalize attention optionally with ReLU
        self.normalization_factor = 1.0 / math.sqrt(self.key_dim)
    
    def build(self, input_shape):
        # Input projections
        self.query_projection = tf.keras.layers.Dense(self.total_key_dim)
        self.key_projection = tf.keras.layers.Dense(self.total_key_dim)
        self.value_projection = tf.keras.layers.Dense(self.total_key_dim)
        self.output_projection = tf.keras.layers.Dense(input_shape[-1])
        
        # Random Fourier Features for approximation
        self.rff = RandomFourierFeatures(
            output_dim=self.feature_dim,
            scale=self.kernel_scale,
            seed=42
        )
        
        # Dropout
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        
    def _split_heads(self, x):
        """Split the channels into multiple heads."""
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.key_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # [B, H, L, K]
    
    def _merge_heads(self, x):
        """Merge the heads back to original shape."""
        batch_size = tf.shape(x)[0]
        x = tf.transpose(x, perm=[0, 2, 1, 3])  # [B, L, H, K]
        return tf.reshape(x, (batch_size, -1, self.total_key_dim))
    
    def _normalize_kernel(self, x):
        """Apply normalization function (softmax or ReLU-based)."""
        if self.use_relu:
            # ReLU-based normalization
            return tf.nn.relu(x) * self.normalization_factor
        else:
            # Exponential-based (softmax-like) normalization
            return tf.exp(x * self.normalization_factor)
    
    def call(self, inputs, mask=None, training=None, bias=None):
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        
        # Project inputs to query, key, value
        q = self.query_projection(inputs)  # [B, L, H*K]
        k = self.key_projection(inputs)    # [B, L, H*K]
        v = self.value_projection(inputs)  # [B, L, H*K]
        
        # Split to multiple heads
        q = self._split_heads(q)  # [B, H, L, K]
        k = self._split_heads(k)  # [B, H, L, K]
        v = self._split_heads(v)  # [B, H, L, K]
        
        # Apply random feature maps
        q_prime = self.rff(q * self.normalization_factor)  # [B, H, L, F]
        k_prime = self.rff(k * self.normalization_factor)  # [B, H, L, F]
        
        if self.causal:
            # For causal masking, we need to do things a bit differently
            output = tf.zeros_like(v)
            cumulative_k = tf.zeros_like(k_prime[:, :, 0:1, :])
            cumulative_kv = tf.zeros_like(v[:, :, 0:1, :])
            
            for i in range(seq_length):
                # Get current position
                q_i = q_prime[:, :, i:i+1, :]  # [B, H, 1, F]
                k_i = k_prime[:, :, i:i+1, :]  # [B, H, 1, F]
                v_i = v[:, :, i:i+1, :]        # [B, H, 1, K]
                
                # Update cumulative tensors
                cumulative_k += k_i
                cumulative_kv += k_i * v_i
                
                # Compute attention output for position i
                output_i = cumulative_kv / (cumulative_k + 1e-6)
        
                # Write to output
                indices = tf.constant([[i]])
                output = tf.tensor_scatter_nd_update(output, indices, output_i)
        else:
            # Non-causal attention with linear complexity
            kv = tf.einsum('bhsf,bhsk->bhfk', k_prime, v)  # [B, H, F, K]
            
            # Denominator for normalization
            k_sum = tf.reduce_sum(k_prime, axis=2, keepdims=True)  # [B, H, 1, F]
            
            # Compute attention
            output = tf.einsum('bhsf,bhfk->bhsk', q_prime, kv)  # [B, H, S, K]
            
            # Normalize
            output = output / (tf.einsum('bhsf,bhf->bhs', q_prime, k_sum[:, :, 0, :]) + 1e-6)[:, :, :, None]
        
        # Merge heads back
        output = self._merge_heads(output)  # [B, S, H*K]
        
        # Apply dropout
        output = self.dropout(output, training=training)
        
        # Project back to input dimension
        output = self.output_projection(output)  # [B, S, D]
        
        return output
    
class Performer(layers.Layer):
    """
    Performer attention layer as a drop-in replacement for MultiHeadAttention.
    """
    def __init__(self, num_heads, key_dim, feature_dim=256, dropout=0.0, 
                 causal=False, use_relu=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.causal = causal
        self.use_relu = use_relu
        
    def build(self, input_shape):
        self.attention = FastSelfAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            feature_dim=self.feature_dim,
            dropout=self.dropout,
            causal=self.causal,
            use_relu=self.use_relu
        )
        
    def call(self, inputs, context=None, training=None, bias=None):
        # For compatibility with MultiHeadAttention interface
        if context is not None:
            inputs = context
            
        return self.attention(inputs, training=training, bias=bias) 