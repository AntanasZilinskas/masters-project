"""
Diffusion-based Oversampling for EVEREST

This module provides a simple diffusion-based model for generating synthetic 
samples of rare events (flares) for oversampling the minority class, as
specified in the RET-plus specification.
"""

import tensorflow as tf
import numpy as np
import os

class DiffusionOversampler:
    """
    Diffusion-based oversampling for generating synthetic minority samples.
    
    This class implements a simple diffusion model (DDPM) for generating
    synthetic samples of rare events, which can be used to balance the
    dataset during training.
    """
    
    def __init__(self, 
                 input_shape,           # Shape of input data (seq_len, features)
                 diffusion_steps=50,    # Number of diffusion steps
                 beta_start=1e-4,       # Initial noise level
                 beta_end=0.02,         # Final noise level
                 model_path=None):      # Path to save/load model
        """
        Initialize the diffusion oversampler.
        
        Args:
            input_shape: Shape of individual samples (seq_len, features)
            diffusion_steps: Number of steps in the diffusion process
            beta_start: Starting value for noise schedule
            beta_end: Ending value for noise schedule
            model_path: Path to save/load the model weights
        """
        self.input_shape = input_shape
        self.diffusion_steps = diffusion_steps
        self.model_path = model_path
        
        # Initialize noise schedule
        self.betas = np.linspace(beta_start, beta_end, diffusion_steps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        
        # Initialize model and sample cache
        self.model = self._build_model()
        self.sample_cache = None
        
    def _build_model(self):
        """Build the diffusion denoising model."""
        # Use a simple U-Net-like architecture for denoising
        inp = tf.keras.layers.Input(shape=self.input_shape)
        time_emb = tf.keras.layers.Input(shape=())
        
        # Convert time embedding to vector
        t_emb = tf.one_hot(tf.cast(time_emb, tf.int32), self.diffusion_steps)
        t_emb = tf.keras.layers.Dense(32, activation='swish')(t_emb)
        t_emb = tf.keras.layers.Dense(64, activation='swish')(t_emb)
        
        # Reshape time embedding to add to each timestep
        t_emb = tf.expand_dims(t_emb, 1)
        t_emb = tf.tile(t_emb, [1, self.input_shape[0], 1])
        
        # Encoder
        x = tf.keras.layers.Conv1D(32, 3, padding='same', activation='swish')(inp)
        x = tf.keras.layers.Concatenate()([x, t_emb])
        x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='swish')(x)
        
        skip1 = x
        x = tf.keras.layers.MaxPooling1D(2)(x)
        
        x = tf.keras.layers.Conv1D(128, 3, padding='same', activation='swish')(x)
        x = tf.keras.layers.Conv1D(128, 3, padding='same', activation='swish')(x)
        
        # Central blocks
        x = tf.keras.layers.Conv1D(128, 3, padding='same', activation='swish')(x)
        x = tf.keras.layers.Conv1D(128, 3, padding='same', activation='swish')(x)
        
        # Decoder with skip connections
        x = tf.keras.layers.UpSampling1D(2)(x)
        x = tf.keras.layers.Concatenate()([x, skip1])
        
        x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='swish')(x)
        x = tf.keras.layers.Conv1D(32, 3, padding='same', activation='swish')(x)
        
        # Output projection
        out = tf.keras.layers.Conv1D(self.input_shape[1], 1, padding='same')(x)
        
        model = tf.keras.Model(inputs=[inp, time_emb], outputs=out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='mse'
        )
        
        return model
    
    def forward_diffusion(self, x_0, t):
        """
        Apply forward diffusion process for timestep t.
        
        Args:
            x_0: Clean data
            t: Timestep
            
        Returns:
            Noisy data x_t and noise
        """
        # Add appropriate level of noise based on timestep
        batch_size = tf.shape(x_0)[0]
        noise = tf.random.normal(shape=tf.shape(x_0))
        
        # Gather alpha based on timestep
        alpha_cumprod_t = tf.gather(self.alphas_cumprod, t)
        alpha_cumprod_t = tf.reshape(alpha_cumprod_t, [batch_size, 1, 1])
        
        # Add noise based on schedule
        x_t = tf.sqrt(alpha_cumprod_t) * x_0 + tf.sqrt(1 - alpha_cumprod_t) * noise
        
        return x_t, noise
    
    def train(self, positive_samples, epochs=50, batch_size=16, validation_split=0.1):
        """
        Train the diffusion model on positive samples.
        
        Args:
            positive_samples: Array of positive class samples
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
        """
        # Convert to float32
        positive_samples = np.array(positive_samples, dtype=np.float32)
        
        # Training loop
        num_samples = len(positive_samples)
        num_batches = num_samples // batch_size
        
        print(f"Training diffusion model on {num_samples} positive samples...")
        
        # Store training history
        loss_history = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(num_samples)
            x_train = positive_samples[indices]
            
            # Initialize epoch loss
            epoch_loss = 0.0
            
            # Process batches
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                x_batch = x_train[start_idx:end_idx]
                
                # Sample timesteps
                t = tf.random.uniform(
                    shape=[end_idx - start_idx],
                    minval=0,
                    maxval=self.diffusion_steps,
                    dtype=tf.int32
                )
                
                # Apply forward diffusion
                x_noisy, noise = self.forward_diffusion(x_batch, t)
                
                # Train step
                loss = self.model.train_on_batch([x_noisy, t], noise)
                epoch_loss += loss
                
            # Average epoch loss
            avg_loss = epoch_loss / num_batches
            loss_history.append(avg_loss)
            
            # Print progress
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Save the model
        if self.model_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save_weights(self.model_path)
            print(f"Model saved to {self.model_path}")
                
        return loss_history
    
    def sample(self, num_samples, temperature=1.0, cache_samples=True):
        """
        Generate synthetic samples using the diffusion model.
        
        Args:
            num_samples: Number of samples to generate
            temperature: Temperature parameter for sampling (higher = more diverse)
            cache_samples: Whether to cache samples for future use
            
        Returns:
            Array of synthetic samples
        """
        # If we have a cache of samples and enough samples, return from cache
        if self.sample_cache is not None and len(self.sample_cache) >= num_samples:
            indices = np.random.choice(len(self.sample_cache), num_samples, replace=False)
            return self.sample_cache[indices]
        
        # Start with random noise
        x_t = tf.random.normal(
            shape=[num_samples, self.input_shape[0], self.input_shape[1]]
        ) * temperature
        
        # Progressively denoise
        for t in range(self.diffusion_steps - 1, -1, -1):
            # Create batch of same timestep
            timesteps = tf.ones(num_samples, dtype=tf.int32) * t
            
            # Predict noise
            predicted_noise = self.model.predict([x_t, timesteps], verbose=0)
            
            # Get alpha values for current timestep
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            
            # Previous alpha for t-1 (use 1.0 for t=0)
            alpha_cumprod_prev = self.alphas_cumprod[t-1] if t > 0 else 1.0
            
            # Calculate coefficients
            beta = 1 - alpha
            
            # Denoise step
            # Only add noise if t > 0
            if t > 0:
                noise = tf.random.normal(shape=tf.shape(x_t)) * temperature
            else:
                noise = 0
            
            # Update x_t
            x_t = (1 / tf.sqrt(alpha)) * (
                x_t - (beta / tf.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + tf.sqrt(beta) * noise
            
            # Optional: Print progress for longer runs
            if self.diffusion_steps > 20 and t % 10 == 0:
                print(f"Sampling step {self.diffusion_steps - t}/{self.diffusion_steps}")
        
        # Convert to numpy array
        synthetic_samples = x_t.numpy()
        
        # Cache samples if requested
        if cache_samples:
            if self.sample_cache is None:
                self.sample_cache = synthetic_samples
            else:
                # Combine with existing cache, but limit total size
                max_cache_size = 5000  # Adjust based on memory constraints
                combined = np.concatenate([self.sample_cache, synthetic_samples], axis=0)
                if len(combined) > max_cache_size:
                    # Randomly subsample to keep cache size reasonable
                    indices = np.random.choice(len(combined), max_cache_size, replace=False)
                    self.sample_cache = combined[indices]
                else:
                    self.sample_cache = combined
            
        return synthetic_samples
    
    def load_model(self, path=None):
        """
        Load model weights from file.
        
        Args:
            path: Path to model weights (if None, uses self.model_path)
        """
        load_path = path or self.model_path
        if load_path and os.path.exists(load_path):
            self.model.load_weights(load_path)
            print(f"Model loaded from {load_path}")
            return True
        else:
            print(f"Model file not found at {load_path}")
            return False

class SMOTEOversampler:
    """
    SMOTE (Synthetic Minority Over-sampling Technique) implementation.
    
    This class provides a simpler alternative to diffusion-based oversampling
    using SMOTE, which generates synthetic samples by interpolating between
    existing minority class examples.
    """
    
    def __init__(self, k_neighbors=5):
        """
        Initialize the SMOTE oversampler.
        
        Args:
            k_neighbors: Number of nearest neighbors to use for interpolation
        """
        self.k_neighbors = k_neighbors
        self.sample_cache = None
        
    def generate_samples(self, positive_samples, num_samples, cache_samples=True):
        """
        Generate synthetic samples using SMOTE.
        
        Args:
            positive_samples: Array of positive class samples
            num_samples: Number of samples to generate
            cache_samples: Whether to cache samples for future use
            
        Returns:
            Array of synthetic samples
        """
        # If we have a cache of samples and enough samples, return from cache
        if self.sample_cache is not None and len(self.sample_cache) >= num_samples:
            indices = np.random.choice(len(self.sample_cache), num_samples, replace=False)
            return self.sample_cache[indices]
        
        # Convert to numpy array
        positive_samples = np.array(positive_samples)
        num_positives = len(positive_samples)
        
        if num_positives < 2:
            raise ValueError("At least 2 positive samples are required for SMOTE")
        
        # Get shape of samples
        sample_shape = positive_samples.shape[1:]
        
        # Flatten samples for easier distance calculation
        flat_samples = positive_samples.reshape(num_positives, -1)
        
        # Create synthetic samples
        synthetic_samples = []
        
        for _ in range(num_samples):
            # Randomly select a positive sample
            i = np.random.randint(0, num_positives)
            
            # Calculate distances to all other samples
            distances = np.sum((flat_samples - flat_samples[i])**2, axis=1)
            
            # Set the distance to self to infinity
            distances[i] = np.inf
            
            # Get k nearest neighbors indices
            k = min(self.k_neighbors, num_positives - 1)  # Ensure k is valid
            nn_indices = np.argsort(distances)[:k]
            
            # Choose one of the neighbors randomly
            nn_idx = np.random.choice(nn_indices)
            
            # Generate synthetic sample by interpolation
            gap = np.random.random()  # Random point between 0 and 1
            synthetic = flat_samples[i] + gap * (flat_samples[nn_idx] - flat_samples[i])
            
            # Reshape back to original sample shape
            synthetic = synthetic.reshape(sample_shape)
            
            synthetic_samples.append(synthetic)
        
        # Convert to numpy array
        synthetic_samples = np.array(synthetic_samples)
        
        # Cache samples if requested
        if cache_samples:
            if self.sample_cache is None:
                self.sample_cache = synthetic_samples
            else:
                # Combine with existing cache, but limit total size
                max_cache_size = 5000
                combined = np.concatenate([self.sample_cache, synthetic_samples], axis=0)
                if len(combined) > max_cache_size:
                    indices = np.random.choice(len(combined), max_cache_size, replace=False)
                    self.sample_cache = combined[indices]
                else:
                    self.sample_cache = combined
        
        return synthetic_samples 