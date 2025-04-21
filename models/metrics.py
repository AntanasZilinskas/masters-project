"""
Custom metrics for solar flare prediction tasks.

This module contains custom metrics specifically designed for evaluating
the performance of solar flare prediction models, particularly focusing
on the True Skill Statistic (TSS) which is a standard metric in space weather forecasting.
"""

import tensorflow as tf

class CategoricalTSSMetric(tf.keras.metrics.Metric):
    """
    True Skill Statistic (TSS) metric for categorical (one-hot encoded) predictions.
    
    TSS = Sensitivity + Specificity - 1
    
    This is a performance metric widely used in solar flare prediction that balances
    the correct prediction rates for both positive and negative classes.
    TSS ranges from -1 to 1, with 1 being perfect prediction, 0 random prediction,
    and negative values worse than random.
    """
    
    def __init__(self, class_id=1, name="tss", **kwargs):
        """
        Initialize the TSS metric.
        
        Args:
            class_id: Integer class ID to calculate metrics for in multi-class case
            name: Name of the metric
            **kwargs: Additional keyword arguments passed to the parent class
        """
        super().__init__(name=name, **kwargs)
        self.class_id = class_id
        self.tp = self.add_weight("tp", initializer="zeros")
        self.tn = self.add_weight("tn", initializer="zeros")
        self.fp = self.add_weight("fp", initializer="zeros")
        self.fn = self.add_weight("fn", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the metric state with batch results.
        
        Args:
            y_true: One-hot encoded true labels
            y_pred: Model predictions (logits or probabilities)
            sample_weight: Optional sample weights
        """
        # Handle both one-hot and logits/probabilistic predictions
        if tf.rank(y_pred) > 1 and tf.shape(y_pred)[-1] > 1:
            # Convert one-hot true labels to class indices for the target class
            y_true_class = tf.cast(tf.equal(tf.argmax(y_true, axis=-1), self.class_id), tf.bool)
            
            # For predictions, get probability of target class
            if tf.keras.backend.is_sparse(y_pred):
                y_pred_probs = tf.nn.softmax(y_pred, axis=-1)[:, self.class_id]
            else:
                y_pred_probs = y_pred[:, self.class_id]
            
            # Convert to binary prediction (>0.5)
            y_pred_class = tf.cast(y_pred_probs > 0.5, tf.bool)
        else:
            # Binary case
            y_true_class = tf.cast(y_true > 0.5, tf.bool)
            y_pred_class = tf.cast(y_pred > 0.5, tf.bool)
        
        # Calculate confusion matrix components
        self.tp.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(y_true_class, y_pred_class), tf.float32)))
        self.tn.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_true_class), tf.logical_not(y_pred_class)), tf.float32)))
        self.fp.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_true_class), y_pred_class), tf.float32)))
        self.fn.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(y_true_class, tf.logical_not(y_pred_class)), tf.float32)))
    
    def result(self):
        """
        Calculate the TSS from accumulated confusion matrix values.
        
        Returns:
            TSS value as a scalar tensor
        """
        sensitivity = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        specificity = self.tn / (self.tn + self.fp + tf.keras.backend.epsilon())
        return sensitivity + specificity - 1
    
    def reset_state(self):
        """Reset all metric state variables."""
        for v in self.variables:
            v.assign(0.0) 