"""
Retentive Layer implementation for EVEREST

This module provides an implementation of the retentive mechanism as described
in the RET-plus specification. It maintains an exponentially decaying state
over time, allowing the model to retain information with a half-life parameter.
"""

import tensorflow as tf
from tensorflow.keras import layers

class RetentiveLayer(layers.Layer):
    """
    Layer that maintains an exponentially decaying state for memory retention.
    
    The layer combines the current hidden state with an exponentially decaying 
    state vector that accumulates information over time:
    
    s_t = λ * s_{t-1} + (1-λ) * h_t
    output = proj([h_t; s_t])
    
    Where:
    - s_t is the state at time t
    - h_t is the input hidden state at time t
    - λ (lambda) is the decay factor (closer to 1 means longer retention)
    """
    
    def __init__(self, output_dim=None, decay_factor=0.95, trainable_decay=True, **kwargs):
        """
        Initialize the retentive layer.
        
        Args:
            output_dim: Dimension of the output projection. If None, will match input dim.
            decay_factor: Initial value of λ (lambda), controls state decay rate.
            trainable_decay: Whether to make the decay factor trainable.
        """
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.initial_decay = decay_factor
        self.trainable_decay = trainable_decay
        
    def build(self, input_shape):
        # Get feature dimension from input shape
        input_dim = input_shape[-1]
        
        # Set output dimension if not specified
        if self.output_dim is None:
            self.output_dim = input_dim
            
        # Create trainable decay parameter (lambda)
        self.decay = self.add_weight(
            name="decay_factor",
            shape=[],
            initializer=tf.constant_initializer(self.initial_decay),
            trainable=self.trainable_decay,
            constraint=lambda x: tf.clip_by_value(x, 0.5, 0.999)  # Constrain between 0.5 and 0.999
        )
        
        # Create the state variable (initialized to zeros)
        self.state = self.add_weight(
            name="memory_state", 
            shape=[input_dim],
            initializer="zeros",
            trainable=False
        )
        
        # Projection layer to convert concatenated [h_t; s_t] to output
        self.projection = layers.Dense(
            self.output_dim,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name="state_projection"
        )
        
    def reset_state(self):
        """Reset the memory state to zeros."""
        self.state.assign(tf.zeros_like(self.state))
        
    def call(self, inputs, training=False):
        # Get batch size and sequence length
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        
        # Store original input shape for reshaping output
        original_shape = tf.shape(inputs)
        
        # Compute the retention outputs for the sequence
        outputs = tf.TensorArray(inputs.dtype, size=seq_length)
        
        # Process the sequence step by step
        for t in range(seq_length):
            # Get the current timestep's hidden state
            h_t = inputs[:, t, :]
            
            # Compute batch-averaged hidden state
            h_avg = tf.reduce_mean(h_t, axis=0)
            
            # Update the state with exponential decay
            # s_t = λ * s_{t-1} + (1-λ) * h_t
            self.state.assign(
                self.decay * self.state + (1.0 - self.decay) * h_avg
            )
            
            # Broadcast state to batch dimension
            state_broadcast = tf.tile(
                tf.expand_dims(self.state, 0), 
                [batch_size, 1]
            )
            
            # Concatenate current hidden state with memory state
            h_with_state = tf.concat([h_t, state_broadcast], axis=-1)
            
            # Project to output dimension
            output_t = self.projection(h_with_state)
            
            # Write to output array
            outputs = outputs.write(t, output_t)
        
        # Stack outputs into a tensor
        outputs = outputs.stack()
        
        # Transpose to [batch_size, seq_length, output_dim]
        outputs = tf.transpose(outputs, [1, 0, 2])
        
        return outputs
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "decay_factor": self.initial_decay,
            "trainable_decay": self.trainable_decay
        })
        return config 