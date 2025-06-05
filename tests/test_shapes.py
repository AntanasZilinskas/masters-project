"""
Tests to ensure model outputs have the correct shapes and types.
"""

import tensorflow as tf


def create_simple_model():
    """Create a simple model for testing."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=(100, 14)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def test_model_output_shape():
    """Test that model output shape is correct (batch, 2)."""
    model = create_simple_model()
    batch_size = 16
    x = tf.random.normal((batch_size, 100, 14))

    output = model(x)

    assert output.shape == (
        batch_size,
        2,
    ), f"Expected shape (batch_size, 2), got {output.shape}"


def test_model_output_dtype():
    """Test that model output dtype is float32."""
    model = create_simple_model()
    batch_size = 16
    x = tf.random.normal((batch_size, 100, 14))

    output = model(x)

    assert output.dtype == tf.float32, f"Expected dtype float32, got {output.dtype}"


def test_model_output_sum_to_one():
    """Test that model outputs sum to 1 on each row (as they're softmax probabilities)."""
    model = create_simple_model()
    batch_size = 16
    x = tf.random.normal((batch_size, 100, 14))

    output = model(x)
    row_sums = tf.reduce_sum(output, axis=1)

    # Check all rows sum to approximately 1
    assert tf.reduce_all(
        tf.abs(row_sums - 1.0) < 1e-5
    ), f"Expected all rows to sum to 1, got: {row_sums}"
