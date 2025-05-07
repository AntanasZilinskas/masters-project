"""
Class-Balanced Focal Loss implementation for EVEREST

This module provides an implementation of Class-Balanced Focal Loss as described
in the paper "Class-Balanced Loss Based on Effective Number of Samples" 
(https://arxiv.org/abs/1901.05555) combined with Focal Loss from 
"Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002).

This loss helps with extremely imbalanced datasets like solar flare prediction.
"""

import tensorflow as tf
import numpy as np

class ClassBalancedFocalLoss(tf.keras.losses.Loss):
    """
    Class-Balanced Focal Loss for better handling of imbalanced datasets.
    
    This loss combines two key ideas:
    1. Class-balanced weighting based on effective number of samples
    2. Focal loss to focus on hard examples
    
    The formula is:
    CB-FL = -α_t * (1-p_t)^γ * log(p_t)
    
    where:
    - α_t is the class weight for target t: (1-β)/(1-β^n_t)
    - p_t is the model's estimated probability for the target class
    - γ is the focusing parameter (higher values focus more on hard examples)
    - n_t is the number of samples in class t
    """
    
    def __init__(self, 
                 beta=0.9999,           # Balance factor (closer to 1 gives more weight to rare classes)
                 gamma=2.0,             # Focusing parameter
                 class_counts=None,     # Number of samples per class
                 from_logits=True,      # Whether input is logits (True) or probabilities (False)
                 label_smoothing=0.0,   # Label smoothing factor
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='cb_focal_loss',
                 **kwargs):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.beta = beta
        self.gamma = gamma
        self.class_counts = class_counts
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        
        # Pre-compute class weights if class_counts is provided
        if class_counts is not None:
            self.class_weights = self._compute_class_weights(class_counts)
        else:
            self.class_weights = None
            
    def _compute_class_weights(self, class_counts):
        """Compute class weights based on effective number of samples."""
        if isinstance(class_counts, (list, tuple)):
            class_counts = np.array(class_counts)
            
        # Calculate effective number of samples
        effective_num = 1.0 - np.power(self.beta, class_counts)
        
        # Calculate weights
        weights = (1.0 - self.beta) / np.where(effective_num > 0, effective_num, 1e-8)
        
        # Normalize weights
        weights = weights / np.sum(weights) * len(weights)
        
        return tf.convert_to_tensor(weights, dtype=tf.float32)
    
    def call(self, y_true, y_pred):
        """
        Calculate the Class-Balanced Focal Loss.
        
        Args:
            y_true: Ground truth labels (one-hot encoded or sparse)
            y_pred: Predicted values (logits or probabilities)
            
        Returns:
            The calculated loss
        """
        # Ensure everything is float32 to avoid type conflicts
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Apply label smoothing if enabled
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)
        
        # Convert to probabilities if from_logits is True
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        # Get batch size
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        
        # Calculate focal weights (1-p_t)^gamma
        # p_t is the probability of the target class
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weights = tf.pow(1.0 - p_t, tf.cast(self.gamma, tf.float32))
        
        # Compute class weighting
        if self.class_weights is not None:
            # Use pre-computed class weights
            # Get the class indices
            class_indices = tf.argmax(y_true, axis=-1)
            # Gather the class weights
            sample_weights = tf.gather(self.class_weights, class_indices)
        else:
            # Compute class weights on-the-fly based on batch statistics
            # Count instances of each class in the batch
            batch_counts = tf.reduce_sum(y_true, axis=0)  # Sum over batch dimension
            batch_counts = tf.cast(batch_counts, tf.float32)
            
            # Apply the effective number formula
            beta_tensor = tf.constant(self.beta, dtype=tf.float32)
            effective_num = 1.0 - tf.pow(beta_tensor, batch_counts)
            
            # Avoid division by zero
            safe_effective_num = tf.where(effective_num > 0, 
                                         effective_num, 
                                         tf.ones_like(effective_num) * 1e-8)
            
            # Calculate weights
            batch_weights = (1.0 - beta_tensor) / safe_effective_num
            
            # Normalize weights
            weight_sum = tf.reduce_sum(batch_weights) + 1e-8
            num_classes = tf.cast(tf.shape(batch_weights)[0], tf.float32)
            batch_weights = batch_weights / weight_sum * num_classes
            
            # Apply to each sample based on its class
            sample_weights = tf.reduce_sum(y_true * tf.expand_dims(batch_weights, 0), axis=-1)
        
        # Apply both class weighting and focal weighting
        weights = sample_weights * focal_weights
        
        # Standard cross-entropy calculation
        # Add a small epsilon to avoid log(0)
        epsilon = 1e-7
        ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred + epsilon), axis=-1)
        
        # Apply the weights
        weighted_loss = weights * ce_loss
        
        # Reduce to scalar
        return tf.reduce_mean(weighted_loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "beta": self.beta,
            "gamma": self.gamma,
            "class_counts": self.class_counts,
            "from_logits": self.from_logits,
            "label_smoothing": self.label_smoothing
        })
        return config

def cb_focal_loss(beta=0.9999, gamma=2.0, class_counts=None, from_logits=True, label_smoothing=0.0):
    """
    Function-based interface for Class-Balanced Focal Loss.
    
    Args:
        beta: Balance factor for class weights (closer to 1 gives more weight to rare classes)
        gamma: Focusing parameter (higher values focus more on hard examples)
        class_counts: Number of samples per class (if None, will be computed from batch)
        from_logits: Whether inputs are logits (True) or probabilities (False)
        label_smoothing: Label smoothing factor
        
    Returns:
        A loss function that can be used with Keras models
    """
    def loss_fn(y_true, y_pred):
        loss = ClassBalancedFocalLoss(
            beta=beta,
            gamma=gamma,
            class_counts=class_counts,
            from_logits=from_logits,
            label_smoothing=label_smoothing
        )
        return loss(y_true, y_pred)
    
    return loss_fn 