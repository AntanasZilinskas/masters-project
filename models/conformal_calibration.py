"""
Conformal Prediction Calibration for EVEREST

This module provides conformal prediction calibration for EVEREST model,
enabling better uncertainty quantification and providing prediction sets
with guaranteed coverage.
"""

import numpy as np
import tensorflow as tf
from sklearn.isotonic import IsotonicRegression

class ConformalCalibrator:
    """
    Conformal calibration wrapper for probabilistic predictions.
    
    This class implements distribution-free conformal prediction, which provides
    prediction sets with guaranteed coverage (1-alpha) regardless of the
    underlying distribution.
    """
    
    def __init__(self, alpha=0.1, isotonic=True):
        """
        Initialize the conformal calibrator.
        
        Args:
            alpha: Significance level (e.g., 0.1 for 90% coverage)
            isotonic: Whether to use isotonic regression for better calibration
        """
        self.alpha = alpha
        self.isotonic = isotonic
        self.threshold = None
        self.calibrated = False
        self.isotonic_model = IsotonicRegression(out_of_bounds='clip') if isotonic else None
        
    def calibrate(self, probs, y_true, mc_samples=None):
        """
        Calibrate the model using validation data.
        
        Args:
            probs: Predicted probabilities from the model
            y_true: True labels (0 or 1)
            mc_samples: Optional tensor of MC dropout samples [n_samples, n_mc, n_classes]
                       for uncertainty-aware calibration
                       
        Returns:
            Calibration threshold
        """
        # If we have MC samples, use them for better calibration
        if mc_samples is not None:
            # Calculate mean and standard deviation of predictions
            mean_probs = np.mean(mc_samples, axis=1)
            std_probs = np.std(mc_samples, axis=1)
            
            # Adjust probabilities based on uncertainty
            adjusted_probs = mean_probs - self.alpha * std_probs
            
            # Use adjusted probabilities for calibration
            probs = adjusted_probs
        
        # Compute non-conformity scores
        # For a binary classifier, the non-conformity score is:
        # - 1 - p(y_i) for the true class
        nonconf_scores = np.zeros_like(y_true, dtype=np.float32)
        
        for i in range(len(y_true)):
            if y_true[i] == 1:
                # For positive class, score is 1 - p(positive)
                nonconf_scores[i] = 1 - probs[i]
            else:
                # For negative class, score is 1 - p(negative) = p(positive)
                nonconf_scores[i] = probs[i]
        
        # Compute the (1-alpha) quantile of non-conformity scores
        self.threshold = np.quantile(nonconf_scores, 1 - self.alpha, interpolation='higher')
        
        # If using isotonic regression, fit it on the calibration data
        if self.isotonic:
            # Only fit on values that make sense (predict positive when true class is positive)
            positive_indices = np.where(y_true == 1)[0]
            if len(positive_indices) > 0:
                pos_probs = probs[positive_indices]
                pos_true = y_true[positive_indices]
                
                # Fit isotonic regression to positive probabilities
                try:
                    self.isotonic_model.fit(pos_probs, pos_true)
                except Exception as e:
                    print(f"Warning: Could not fit isotonic regression: {e}")
                    self.isotonic = False
        
        self.calibrated = True
        return self.threshold
        
    def predict_sets(self, probs, mc_samples=None):
        """
        Generate prediction sets using conformal calibration.
        
        Args:
            probs: Predicted probabilities from the model
            mc_samples: Optional tensor of MC dropout samples for better calibration
            
        Returns:
            A dictionary containing:
            - 'sets': Boolean array where True indicates the class is in the prediction set
            - 'lower': Lower bound of prediction interval
            - 'upper': Upper bound of prediction interval
            - 'point': Point estimate (highest probability class)
        """
        if not self.calibrated:
            raise ValueError("Calibrate the model first by calling calibrate()")
        
        # If we have MC samples, use them for uncertainty-aware predictions
        if mc_samples is not None:
            # Calculate mean and std of probabilities
            mean_probs = np.mean(mc_samples, axis=1)
            std_probs = np.std(mc_samples, axis=1)
            
            # Adjust probabilities based on uncertainty
            # (more uncertainty → wider prediction sets)
            adjusted_probs = mean_probs - self.alpha * std_probs
            
            # Using adjusted probabilities for set construction
            probs_for_sets = adjusted_probs
        else:
            probs_for_sets = probs
        
        # Apply isotonic regression if enabled
        if self.isotonic and self.isotonic_model is not None:
            try:
                # Transform probabilities using isotonic regression
                probs_for_sets = self.isotonic_model.transform(probs_for_sets)
            except Exception as e:
                print(f"Warning: Error applying isotonic regression: {e}")
        
        # Construct prediction sets
        # A class is in the set if 1 - p(class) ≤ threshold
        # For binary classification, we only need to check the positive class
        positive_in_set = 1 - probs_for_sets <= self.threshold
        
        # Create point predictions (highest probability class)
        point_predictions = (probs > 0.5).astype(int)
        
        # Calculate lower and upper bounds for uncertainty intervals
        if mc_samples is not None:
            # Using Monte Carlo samples for intervals
            lower = np.percentile(mc_samples, self.alpha / 2 * 100, axis=1)
            upper = np.percentile(mc_samples, (1 - self.alpha / 2) * 100, axis=1)
        else:
            # Simple interval based on threshold
            # This is less informative without MC samples
            lower = np.where(positive_in_set, probs_for_sets, 0)
            upper = np.where(positive_in_set, 1, probs_for_sets)
        
        return {
            'sets': positive_in_set,  # Boolean array (True = positive class in set)
            'lower': lower,           # Lower bound of prediction interval
            'upper': upper,           # Upper bound of prediction interval
            'point': point_predictions  # Point estimate (highest probability class)
        }
    
    def evaluate(self, probs, y_true, mc_samples=None):
        """
        Evaluate the calibration on a test set.
        
        Args:
            probs: Predicted probabilities from the model
            y_true: True labels (0 or 1)
            mc_samples: Optional tensor of MC dropout samples
            
        Returns:
            Dictionary of metrics (coverage, efficiency, etc.)
        """
        if not self.calibrated:
            raise ValueError("Calibrate the model first by calling calibrate()")
        
        # Get prediction sets
        pred_sets = self.predict_sets(probs, mc_samples)
        
        # Calculate metrics
        # Coverage: What fraction of true labels are in the prediction sets?
        # For binary classification: when y=1, the set must contain 1 (positive_in_set=True)
        #                           when y=0, either the set contains 0 or 1 is not in the set
        coverage_positive = np.mean(pred_sets['sets'][y_true == 1])
        coverage_negative = np.mean(~pred_sets['sets'][y_true == 0])
        
        # Overall coverage
        coverage = np.mean((y_true == 1) & pred_sets['sets'] | (y_true == 0) & ~pred_sets['sets'])
        
        # Set size: Average number of classes in the prediction sets
        # For binary, this is just the fraction of sets that include the positive class
        set_size = np.mean(pred_sets['sets'])
        
        # Efficiency: Ideally, sets contain exactly the true class and no others
        # For binary, we calculate the fraction of singleton sets that are correct
        singleton_indices = np.where(pred_sets['sets'] == 1)[0]
        efficiency = np.mean(y_true[singleton_indices] == 1) if len(singleton_indices) > 0 else 0
        
        return {
            'coverage': float(coverage),
            'coverage_positive': float(coverage_positive),
            'coverage_negative': float(coverage_negative),
            'set_size': float(set_size),
            'efficiency': float(efficiency),
            'threshold': float(self.threshold)
        }
    
    def save(self, path):
        """Save the calibration data to a file."""
        save_data = {
            'alpha': self.alpha,
            'threshold': self.threshold,
            'calibrated': self.calibrated,
            'isotonic': self.isotonic
        }
        
        # Save as numpy file
        np.save(path, save_data)
        
    def load(self, path):
        """Load calibration data from a file."""
        try:
            data = np.load(path, allow_pickle=True).item()
            self.alpha = data['alpha']
            self.threshold = data['threshold']
            self.calibrated = data['calibrated']
            self.isotonic = data['isotonic']
            return True
        except Exception as e:
            print(f"Error loading calibration data: {e}")
            return False

def calibrate_model(model, X_val, y_val, alpha=0.1, mc_samples=20):
    """
    Utility function to calibrate a Keras model with conformal prediction.
    
    Args:
        model: The Keras model to calibrate
        X_val: Validation features
        y_val: Validation labels (0/1)
        alpha: Significance level (e.g., 0.1 for 90% coverage)
        mc_samples: Number of Monte Carlo dropout samples to use
        
    Returns:
        Calibrated model and conformal calibrator
    """
    # Generate Monte Carlo samples
    mc_preds = []
    for _ in range(mc_samples):
        # Use training=True to activate dropout during inference
        preds = model(X_val, training=True)
        if isinstance(preds, dict):
            # For EVEREST with multiple outputs, use softmax_dense
            preds = preds["softmax_dense"]
        mc_preds.append(preds.numpy() if hasattr(preds, 'numpy') else preds)
    
    # Stack samples
    mc_preds = np.stack(mc_preds, axis=1)  # [n_samples, n_mc, n_classes]
    
    # Calculate mean predictions
    mean_preds = np.mean(mc_preds, axis=1)
    if mean_preds.shape[1] == 2:  # If one-hot encoded
        mean_preds = mean_preds[:, 1]  # Take probability of positive class
        
    # Create and calibrate
    calibrator = ConformalCalibrator(alpha=alpha)
    calibrator.calibrate(mean_preds, y_val, mc_preds)
    
    # Return the calibrated wrapper
    return calibrator 