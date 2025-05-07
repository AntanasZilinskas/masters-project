"""
Diffusion-based oversampling module for EVEREST.

This module implements a 1D DDPM (Denoising Diffusion Probabilistic Model)
for generating synthetic samples of rare events (solar flares).
"""

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers, models

class DiffusionModel(tf.keras.Model):
    def __init__(self, seq_len, n_feat, diffusion_steps=100):
        super().__init__()
        self.seq_len = seq_len
        self.n_feat = n_feat
        self.diffusion_steps = diffusion_steps
        
        # Define beta schedule for variance growth
        self.betas = self._get_beta_schedule()
        self.alphas = 1 - self.betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas)
        
        # Build the denoising UNet model
        self.denoiser = self._build_denoiser()
        
    def _get_beta_schedule(self):
        """Linear beta schedule."""
        beta_start = 1e-4
        beta_end = 2e-2
        betas = np.linspace(beta_start, beta_end, self.diffusion_steps)
        return tf.constant(betas, dtype=tf.float32)
        
    def _build_denoiser(self):
        """Simple 1D CNN for denoising - minimalist version without skip connections."""
        time_embed_dim = 32
        
        # Input layers
        x_input = layers.Input(shape=(self.seq_len, self.n_feat))
        t_input = layers.Input(shape=())
        
        # Time embedding
        t_embed = tf.one_hot(tf.cast(t_input, tf.int32), self.diffusion_steps)
        t_embed = layers.Dense(time_embed_dim, activation="swish")(t_embed)
        t_embed = layers.Dense(time_embed_dim, activation="swish")(t_embed)
        
        # Reshape time embedding to match sequence dimension
        t_embed = tf.reshape(t_embed, [-1, 1, time_embed_dim])
        t_embed = tf.tile(t_embed, [1, self.seq_len, 1])
        
        # Simple feed-forward network - much more stable than U-Net for sequences
        # Expand feature dimension
        x = layers.Conv1D(32, 3, padding="same", activation="swish")(x_input)
        x = layers.BatchNormalization()(x)
        
        # Add time embedding
        x = layers.Concatenate()([x, t_embed])
        
        # Main processing blocks
        for filters in [64, 128, 128, 64]:
            x = layers.Conv1D(filters, 3, padding="same", activation="swish")(x)
            x = layers.BatchNormalization()(x)
        
        # Output projection
        x = layers.Conv1D(self.n_feat, 1, padding="same")(x)
        
        return tf.keras.Model([x_input, t_input], x)
    
    def diffusion_forward(self, x_0, t):
        """Forward diffusion process: q(x_t | x_0)."""
        # Ensure all inputs have the same dtype
        x_0 = tf.cast(x_0, tf.float32)
        
        batch_size = tf.shape(x_0)[0]
        noise = tf.random.normal(shape=tf.shape(x_0), dtype=tf.float32)
        
        # Gather the alphas based on timestep
        alpha_cumprod_t = tf.gather(self.alphas_cumprod, t)
        alpha_cumprod_t = tf.reshape(alpha_cumprod_t, [batch_size, 1, 1])
        alpha_cumprod_t = tf.cast(alpha_cumprod_t, tf.float32)
        
        # Calculate mean and variance for the forward process
        mean = tf.sqrt(alpha_cumprod_t) * x_0
        var = 1 - alpha_cumprod_t
        
        # Apply the forward diffusion formula
        return mean + tf.sqrt(var) * noise
    
    def call(self, inputs, training=False):
        """Predict noise given noisy sample."""
        x, t = inputs
        return self.denoiser([x, t], training=training)
    
    def reverse_diffusion(self, batch_size):
        """Generate new samples by reverse diffusion."""
        # Start from pure noise
        x_t = tf.random.normal(shape=[batch_size, self.seq_len, self.n_feat], dtype=tf.float32)
        
        # Iterate backward
        for t in range(self.diffusion_steps - 1, -1, -1):
            time_tensor = tf.ones(batch_size, dtype=tf.int32) * t
            
            # Predict denoised x_0
            predicted_noise = self.denoiser([x_t, time_tensor], training=False)
            
            # Get parameters for reverse step
            alpha_t = tf.cast(self.alphas[t], tf.float32)
            alpha_cumprod_t = tf.cast(self.alphas_cumprod[t], tf.float32)
            
            # Add noise for t > 0
            noise = tf.random.normal(shape=tf.shape(x_t), dtype=tf.float32) if t > 0 else 0
            
            # Update x_t
            x_t = (1 / tf.sqrt(alpha_t)) * (x_t - (1 - alpha_t) / tf.sqrt(1 - alpha_cumprod_t) * predicted_noise)
            
            if t > 0:
                sigma_t = tf.sqrt(tf.cast(self.betas[t], tf.float32))
                x_t = x_t + sigma_t * noise
                
        return x_t

def train_sampler(X_pos, n_steps=100, save_path="sampler.h5", epochs=50, batch_size=16):
    """
    Train a diffusion model on positive (flare) samples.
    
    Args:
        X_pos: Training data of positive samples, shape (n_samples, seq_len, n_feat)
        n_steps: Number of diffusion steps
        save_path: Path to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        Trained diffusion model
    """
    # Ensure X_pos is float32 to avoid type mismatches
    X_pos = np.array(X_pos, dtype=np.float32)
    
    seq_len, n_feat = X_pos.shape[1], X_pos.shape[2]
    
    # Create the diffusion model
    model = DiffusionModel(seq_len, n_feat, diffusion_steps=n_steps)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=2e-4),
        loss=tf.keras.losses.MeanSquaredError()
    )
    
    # Create TensorFlow dataset for better performance
    dataset = tf.data.Dataset.from_tensor_slices(X_pos)
    dataset = dataset.shuffle(buffer_size=len(X_pos)).batch(batch_size)
    
    # Calculate total number of batches for progress reporting
    total_batches = len(X_pos) // batch_size + (1 if len(X_pos) % batch_size != 0 else 0)
    
    print(f"Training diffusion model for {epochs} epochs, {total_batches} batches per epoch")
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0
        batch_count = 0
        
        # Use tqdm if available for a progress bar, otherwise use simple counter
        try:
            from tqdm import tqdm
            batch_iterator = tqdm(dataset, total=total_batches)
        except ImportError:
            print("Install tqdm for progress bars")
            batch_iterator = dataset
            
        for x_batch in batch_iterator:
            # Ensure batch is float32
            x_batch = tf.cast(x_batch, tf.float32)
            batch_size = tf.shape(x_batch)[0]
            
            # Sample random timesteps
            t = tf.random.uniform(
                shape=[batch_size], 
                minval=0, 
                maxval=n_steps, 
                dtype=tf.int32
            )
            
            # Generate target noise
            noise = tf.random.normal(shape=tf.shape(x_batch), dtype=tf.float32)
            
            # Forward diffusion
            noisy_x = model.diffusion_forward(x_batch, t)
            
            # Train step - predict the noise
            with tf.GradientTape() as tape:
                predicted_noise = model([noisy_x, t], training=True)
                loss = tf.reduce_mean(tf.square(noise - predicted_noise))
            
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            epoch_loss += loss.numpy()
            batch_count += 1
            
            # Print progress without tqdm
            if 'tqdm' not in locals():
                print(f"Batch {batch_count}/{total_batches}, Loss: {loss:.4f}", end="\r")
        
        avg_loss = epoch_loss / batch_count
        print(f"\nEpoch {epoch+1} average loss: {avg_loss:.4f}")
    
    # Save the model
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save_weights(save_path)
        print(f"Model saved to {save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        print("Continuing with in-memory model")
    
    return model

def sample(n, seq_len, n_feat, model_path="sampler.h5", diffusion_steps=100):
    """
    Generate n synthetic sequences using the trained diffusion model.
    
    Args:
        n: Number of sequences to generate
        seq_len: Sequence length
        n_feat: Number of features
        model_path: Path to the saved model weights
        diffusion_steps: Number of diffusion steps (must match training)
    
    Returns:
        Array of synthetic sequences, shape (n, seq_len, n_feat)
    """
    # Create the model
    model = DiffusionModel(seq_len, n_feat, diffusion_steps=diffusion_steps)
    
    # Initialize the model by calling it once with dummy data
    dummy_x = tf.zeros((1, seq_len, n_feat))
    dummy_t = tf.zeros((1,), dtype=tf.int32)
    _ = model([dummy_x, dummy_t])
    
    # Now load the weights (if available)
    if os.path.exists(model_path):
        try:
            model.load_weights(model_path)
            print(f"Successfully loaded diffusion model from {model_path}")
        except Exception as e:
            print(f"Error loading diffusion model weights: {e}")
            print("Using untrained model.")
    else:
        print(f"Warning: Model not found at {model_path}. Using untrained model.")
    
    # Generate samples
    return model.reverse_diffusion(n).numpy() 