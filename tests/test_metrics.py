"""
Tests to ensure the metrics calculations work as expected.
"""

import numpy as np
import tensorflow as tf


class TrueSkillStatistic(tf.keras.metrics.Metric):
    """Custom metric for solar prediction skill."""

    def __init__(self, name="true_skill_statistic", **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.true_negatives = self.add_weight(name="tn", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_binary = tf.argmax(y_pred, axis=1)
        y_true = tf.cast(y_true, dtype=tf.int64)

        # Calculate confusion matrix values
        self.true_positives.assign_add(
            tf.reduce_sum(
                tf.cast(
                    tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred_binary, 1)),
                    tf.float32,
                )
            )
        )
        self.false_positives.assign_add(
            tf.reduce_sum(
                tf.cast(
                    tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred_binary, 1)),
                    tf.float32,
                )
            )
        )
        self.true_negatives.assign_add(
            tf.reduce_sum(
                tf.cast(
                    tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred_binary, 0)),
                    tf.float32,
                )
            )
        )
        self.false_negatives.assign_add(
            tf.reduce_sum(
                tf.cast(
                    tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred_binary, 0)),
                    tf.float32,
                )
            )
        )

    def result(self):
        # Calculate True Skill Statistic (TSS)
        # TSS = (TP/(TP+FN)) - (FP/(FP+TN))
        pod = self.true_positives / (
            self.true_positives + self.false_negatives + tf.keras.backend.epsilon()
        )
        pofd = self.false_positives / (
            self.false_positives + self.true_negatives + tf.keras.backend.epsilon()
        )
        return pod - pofd

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.true_negatives.assign(0.0)
        self.false_negatives.assign(0.0)


def test_tss_perfect_predictions():
    """Test TrueSkillStatistic returns 1.0 on perfect predictions."""
    metric = TrueSkillStatistic()

    # Create perfect predictions
    y_true = tf.constant([0, 1, 0, 1, 0, 1])
    # Binary classification with perfect predictions
    y_pred = tf.constant(
        [
            [1.0, 0.0],  # Class 0
            [0.0, 1.0],  # Class 1
            [1.0, 0.0],  # Class 0
            [0.0, 1.0],  # Class 1
            [1.0, 0.0],  # Class 0
            [0.0, 1.0],  # Class 1
        ]
    )

    metric.update_state(y_true, y_pred)
    result = metric.result().numpy()

    # With perfect predictions, TSS should be 1.0
    assert np.isclose(
        result, 1.0, atol=1e-6
    ), f"Expected TSS=1.0 for perfect predictions, got {result}"


def test_tss_random_predictions():
    """Test TrueSkillStatistic returns 0.0 on random predictions."""
    metric = TrueSkillStatistic()

    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Test with 1000 random samples to get stable statistics
    n_samples = 1000
    y_true = tf.random.uniform((n_samples,), maxval=2, dtype=tf.int32)

    # Create random predictions (50/50 split)
    random_probs = tf.random.uniform((n_samples, 2))
    # Normalize to sum to 1 on rows
    y_pred = random_probs / tf.reduce_sum(random_probs, axis=1, keepdims=True)

    metric.update_state(y_true, y_pred)
    result = metric.result().numpy()

    # With random predictions, TSS should be close to 0.0
    assert np.isclose(
        result, 0.0, atol=0.1
    ), f"Expected TSS≈0.0 for random predictions, got {result}"
