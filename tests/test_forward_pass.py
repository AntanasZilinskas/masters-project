"""
Tests to ensure the model's forward pass works correctly and loss decreases during training.
"""

import tensorflow as tf


def create_training_model():
    """Create a model for training tests."""
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


def test_loss_decreases():
    """Test that loss decreases after training on a few batches."""
    model = create_training_model()

    # Create two mini-batches
    batch_size = 16
    x_train = tf.random.normal((batch_size * 2, 100, 14))
    y_train = tf.random.uniform((batch_size * 2,), maxval=2, dtype=tf.int32)

    # Split into two batches
    x_batch1, x_batch2 = tf.split(x_train, 2)
    y_batch1, y_batch2 = tf.split(y_train, 2)

    # Get initial loss on first batch
    initial_loss = model.evaluate(x_batch1, y_batch1, verbose=0)[0]

    # Train on first batch
    model.fit(x_batch1, y_batch1, epochs=5, verbose=0)

    # Get loss after training
    post_training_loss = model.evaluate(x_batch1, y_batch1, verbose=0)[0]

    # Check that loss decreased
    assert (
        post_training_loss < initial_loss
    ), f"Loss should decrease after training. Initial: {initial_loss}, After: {post_training_loss}"


def test_can_predict():
    """Test that model can make predictions."""
    model = create_training_model()
    batch_size = 8
    x = tf.random.normal((batch_size, 100, 14))

    # Make predictions
    preds = model.predict(x)

    # Check shape and type
    assert preds.shape == (
        batch_size,
        2,
    ), f"Expected shape (batch_size, 2), got {preds.shape}"
    assert isinstance(preds, tf.Tensor) or isinstance(
        preds, tf.numpy.ndarray
    ), f"Expected TensorFlow tensor or numpy array, got {type(preds)}"
