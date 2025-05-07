"""
Metrics for solar flare prediction

This module provides custom metrics for evaluating solar flare prediction models,
including the True Skill Statistic (TSS) which is commonly used in space weather
forecasting applications.
"""

import tensorflow as tf
import numpy as np

class CategoricalTSSMetric(tf.keras.metrics.Metric):
    """
    True Skill Statistic (TSS) metric for categorical data.
    
    TSS = (TP/(TP+FN)) - (FP/(FP+TN)) = sensitivity + specificity - 1
    
    This metric is particularly useful for imbalanced datasets because it is
    insensitive to the class imbalance ratio. It ranges from -1 to 1, where:
    - 1 is perfect prediction
    - 0 is random prediction
    - -1 is perfectly inverse prediction
    """
    
    def __init__(self, name='tss', class_id=1, **kwargs):
        """
        Initialize the TSS metric.
        
        Args:
            name: Name of the metric
            class_id: Index of the positive class (default: 1)
        """
        super().__init__(name=name, **kwargs)
        self.class_id = class_id
        self.true_positives = self.add_weight(
            'true_positives', initializer='zeros')
        self.true_negatives = self.add_weight(
            'true_negatives', initializer='zeros')
        self.false_positives = self.add_weight(
            'false_positives', initializer='zeros')
        self.false_negatives = self.add_weight(
            'false_negatives', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the state of the metric with new data.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            sample_weight: Optional sample weights
        """
        # Get the positive and negative classes
        if isinstance(y_true, tf.RaggedTensor):
            y_true = y_true.to_tensor()
            
        # Convert to dense tensors if needed
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        
        # Ensure y_true is one-hot encoded
        if y_true.shape[-1] == 1 or len(y_true.shape) == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), 2)
        
        # Get the predictions as class indices (argmax)
        pred_class = tf.argmax(y_pred, axis=-1)
        pred_class = tf.one_hot(pred_class, depth=tf.shape(y_pred)[-1])
        
        # Extract the positive and negative classes
        positive_class = tf.cast(y_true[:, self.class_id], tf.float32)
        negative_class = 1.0 - positive_class
        
        pred_positive = tf.cast(pred_class[:, self.class_id], tf.float32)
        pred_negative = 1.0 - pred_positive
        
        # Apply sample weights if provided
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.reshape(sample_weight, (-1, 1))
            positive_class = positive_class * sample_weight
            negative_class = negative_class * sample_weight
            pred_positive = pred_positive * sample_weight
            pred_negative = pred_negative * sample_weight
        
        # Calculate TP, TN, FP, FN
        tp = tf.reduce_sum(positive_class * pred_positive)
        tn = tf.reduce_sum(negative_class * pred_negative)
        fp = tf.reduce_sum(negative_class * pred_positive)
        fn = tf.reduce_sum(positive_class * pred_negative)
        
        # Update the metric state
        self.true_positives.assign_add(tp)
        self.true_negatives.assign_add(tn)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)
    
    def result(self):
        """
        Calculate the TSS from the accumulated statistics.
        
        Returns:
            TSS value
        """
        # Calculate Probability of Detection (POD) / Sensitivity / Recall
        # POD = TP / (TP + FN)
        tp = self.true_positives
        fn = self.false_negatives
        sensitivity = tp / (tp + fn + tf.keras.backend.epsilon())
        
        # Calculate False Alarm Ratio (FAR) / Fall-out / (1 - Specificity)
        # FAR = FP / (FP + TN)
        fp = self.false_positives
        tn = self.true_negatives
        far = fp / (fp + tn + tf.keras.backend.epsilon())
        
        # Calculate TSS
        # TSS = Sensitivity - FAR = Sensitivity + Specificity - 1
        tss = sensitivity - far
        
        return tss
    
    def reset_state(self):
        """Reset the metric state."""
        self.true_positives.assign(0.0)
        self.true_negatives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)

class ThresholdTuner(tf.keras.callbacks.Callback):
    """
    Callback for finding the optimal prediction threshold based on TSS.
    
    This callback evaluates the model on a validation set at the end of each
    epoch, trying different thresholds to find the one that maximizes the TSS.
    """
    
    def __init__(self, validation_data, patience=5, output_name=None):
        """
        Initialize the threshold tuner.
        
        Args:
            validation_data: Tuple of (X_val, y_val)
            patience: Number of epochs to wait for improvement
            output_name: Name of the model output to use (for multi-output models)
        """
        super().__init__()
        self.validation_data = validation_data
        self.patience = patience
        self.output_name = output_name
        self.best_threshold = 0.5
        self.best_tss = -1.0
        self.best_f1 = 0.0
        self.thresholds = np.linspace(0.1, 0.9, 17)
        self.wait = 0
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        logs = logs or {}
        
        # Get validation data
        x_val, y_val = self.validation_data
        
        # Get predictions
        y_pred = self.model.predict(x_val)
        
        # If multi-output model, get the output we're interested in
        if isinstance(y_pred, dict) and self.output_name is not None:
            y_pred = y_pred[self.output_name]
        
        # If one-hot encoded, use positive class probabilities
        if isinstance(y_pred, np.ndarray) and y_pred.shape[-1] > 1:
            y_pred_pos = y_pred[:, 1]
        else:
            y_pred_pos = y_pred.flatten()
            
        # If y_val is one-hot encoded, get binary labels
        if isinstance(y_val, np.ndarray) and y_val.shape[-1] > 1:
            y_val_bin = y_val[:, 1]
        else:
            y_val_bin = y_val.flatten()
            
        # Ensure binary labels
        y_val_bin = (y_val_bin > 0.5).astype(int)
        
        # Find best threshold
        best_tss = -1.0
        best_threshold = 0.5
        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0
        
        # Try different thresholds
        for threshold in self.thresholds:
            preds = (y_pred_pos >= threshold).astype(int)
            
            # Calculate metrics
            tp = np.sum((preds == 1) & (y_val_bin == 1))
            tn = np.sum((preds == 0) & (y_val_bin == 0))
            fp = np.sum((preds == 1) & (y_val_bin == 0))
            fn = np.sum((preds == 0) & (y_val_bin == 1))
            
            # Avoid division by zero
            sensitivity = tp / (tp + fn + 1e-8)
            specificity = tn / (tn + fp + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            recall = sensitivity
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            # Calculate TSS
            tss = sensitivity + specificity - 1.0
            
            # Check if better
            if tss > best_tss:
                best_tss = tss
                best_threshold = threshold
                best_f1 = f1
                best_precision = precision
                best_recall = recall
        
        # Update best if improved
        if best_tss > self.best_tss:
            self.best_tss = best_tss
            self.best_threshold = best_threshold
            self.best_f1 = best_f1
            self.wait = 0
        else:
            self.wait += 1
        
        # Print results
        print(f"\n— prec={best_precision:.4f}  rec={best_recall:.4f}  tss={best_tss:.4f}")
        
        # Add to logs
        logs['best_threshold'] = self.best_threshold
        logs['best_tss'] = self.best_tss
        logs['best_f1'] = self.best_f1
        logs['best_precision'] = best_precision
        logs['best_recall'] = best_recall
        
        return logs
    
def calculate_tss_from_cm(tn, fp, fn, tp):
    """
    Calculate TSS from confusion matrix elements.
    
    Args:
        tn: True negatives
        fp: False positives
        fn: False negatives
        tp: True positives
        
    Returns:
        TSS value
    """
    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    
    # Calculate TSS
    tss = sensitivity + specificity - 1.0
    
    return tss 