"""
Multi-Scale Tokenizer for EVEREST

This module provides a multi-scale tokenizer that processes input sequences at
multiple time scales (10min, 1h, 3h) to capture temporal patterns at different
resolutions, as specified in the RET-plus specification.
"""

import tensorflow as tf

class MultiScaleTokenizer(tf.keras.layers.Layer):
    """
    Multi-Scale Tokenizer that processes input at 10min, 1h, and 3h resolutions.
    
    This layer takes a time series and creates multiple views at different
    temporal resolutions using average pooling, then concatenates them along
    the time dimension to create a multi-scale representation.
    """
    
    def __init__(self, 
                 include_original=True,  # Whether to include the original 10min data
                 include_1h=True,        # Whether to include the 1h average
                 include_3h=True,        # Whether to include the 3h average
                 pooling_method='avg',   # Pooling method: 'avg' or 'max'
                 **kwargs):
        """
        Initialize the Multi-Scale Tokenizer.
        
        Args:
            include_original: Whether to include the original 10min resolution
            include_1h: Whether to include the 1h pooled representation
            include_3h: Whether to include the 3h pooled representation
            pooling_method: Method for pooling ('avg' or 'max')
        """
        super().__init__(**kwargs)
        self.include_original = include_original
        self.include_1h = include_1h
        self.include_3h = include_3h
        self.pooling_method = pooling_method
        
        # Validation
        if not (include_original or include_1h or include_3h):
            raise ValueError("At least one resolution must be included")
            
        if pooling_method not in ['avg', 'max']:
            raise ValueError("pooling_method must be 'avg' or 'max'")
            
    def build(self, input_shape):
        # No weights to build, but we'll validate the input shape
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape (batch_size, seq_len, features), got {input_shape}")
            
        # Create projection layers for each scale to ensure consistent features
        self.feature_dim = input_shape[-1]
        self.proj_1h = tf.keras.layers.Dense(
            self.feature_dim, 
            kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )
        
        self.proj_3h = tf.keras.layers.Dense(
            self.feature_dim, 
            kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )
            
    def call(self, inputs, training=None):
        """
        Process inputs at multiple time scales.
        
        Args:
            inputs: Input tensor of shape [batch_size, seq_len, features]
            
        Returns:
            Multi-scale representation with original concatenated with pooled versions
        """
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Compute 1h pooling (6 timesteps for 10min data)
        # Skip if not including this resolution
        if self.include_1h:
            if self.pooling_method == 'avg':
                # Use AveragePooling1D instead of AvgPool1D
                x_1h = tf.keras.layers.AveragePooling1D(pool_size=6, strides=6, padding='valid')(inputs)
            else:
                # Use max_pool1d on the time dimension
                x_1h = tf.keras.layers.MaxPool1D(pool_size=6, strides=6, padding='valid')(inputs)
                
            # Project to ensure feature consistency
            x_1h = self.proj_1h(x_1h)
            
            # Upsample back to original sequence length
            # We'll use tf.image.resize with nearest neighbor for simplicity
            # First reshape to [batch_size, reduced_seq_len, 1, features]
            x_1h_shape = tf.shape(x_1h)
            x_1h_reshaped = tf.reshape(x_1h, [batch_size, x_1h_shape[1], 1, self.feature_dim])
            
            # Resize to [batch_size, seq_len, 1, features]
            x_1h_upsampled = tf.image.resize(
                x_1h_reshaped, 
                [seq_len, 1], 
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            
            # Reshape back to [batch_size, seq_len, features]
            x_1h = tf.reshape(x_1h_upsampled, [batch_size, seq_len, self.feature_dim])
        
        # Compute 3h pooling (18 timesteps for 10min data)
        # Skip if not including this resolution
        if self.include_3h:
            if self.pooling_method == 'avg':
                # Use AveragePooling1D instead of AvgPool1D
                x_3h = tf.keras.layers.AveragePooling1D(pool_size=18, strides=18, padding='valid')(inputs)
            else:
                # Use max_pool1d on the time dimension
                x_3h = tf.keras.layers.MaxPool1D(pool_size=18, strides=18, padding='valid')(inputs)
                
            # Project to ensure feature consistency
            x_3h = self.proj_3h(x_3h)
            
            # Upsample back to original sequence length
            # We'll use tf.image.resize with nearest neighbor for simplicity
            # First reshape to [batch_size, reduced_seq_len, 1, features]
            x_3h_shape = tf.shape(x_3h)
            x_3h_reshaped = tf.reshape(x_3h, [batch_size, x_3h_shape[1], 1, self.feature_dim])
            
            # Resize to [batch_size, seq_len, 1, features]
            x_3h_upsampled = tf.image.resize(
                x_3h_reshaped, 
                [seq_len, 1], 
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            
            # Reshape back to [batch_size, seq_len, features]
            x_3h = tf.reshape(x_3h_upsampled, [batch_size, seq_len, self.feature_dim])
        
        # Collect all the representations we want to include
        representations = []
        
        if self.include_original:
            representations.append(inputs)
            
        if self.include_1h:
            representations.append(x_1h)
            
        if self.include_3h:
            representations.append(x_3h)
            
        # Concatenate along the feature dimension (last axis)
        result = tf.concat(representations, axis=-1)
        
        return result
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "include_original": self.include_original,
            "include_1h": self.include_1h,
            "include_3h": self.include_3h,
            "pooling_method": self.pooling_method
        })
        return config 