"""Phase 4: same architecture as numpy_model/network.py, in Keras.

    12288 → 128 (ReLU) → 64 (ReLU) → 7 (softmax)

Same data arrays, same mini-batch SGD, same 20 epochs / batch 64 / lr 0.5.
He-normal weights, zero biases — closest Keras match to the NumPy init.
"""

from __future__ import annotations

import time

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from numpy_model.network import LAYER_DIMS


def build_model(
    layer_dims: list[int] = LAYER_DIMS,
    learning_rate: float = 0.5,
    seed: int = 42,
) -> keras.Model:
    """Dense stack matching LAYER_DIMS. SGD, not Adam — that would not be a fair comparison."""
    keras.utils.set_random_seed(seed)
    he = keras.initializers.HeNormal(seed=seed)

    model = keras.Sequential(name="car_mlp")
    model.add(keras.Input(shape=(layer_dims[0],)))
    for n_out in layer_dims[1:-1]:
        model.add(
            layers.Dense(
                n_out,
                activation="relu",
                kernel_initializer=he,
                bias_initializer="zeros",
            )
        )
    model.add(
        layers.Dense(
            layer_dims[-1],
            activation="softmax",
            kernel_initializer=he,
            bias_initializer="zeros",
        )
    )
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.5,
    seed: int = 42,
) -> tuple[keras.Model, dict, float]:
    """Fit under the same conditions as numpy_model.network.train.

    Returns (model, keras history.history dict, wall-clock seconds).
    """
    model = build_model(learning_rate=learning_rate, seed=seed)
    t0 = time.perf_counter()
    hist = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
    )
    elapsed = time.perf_counter() - t0
    return model, dict(hist.history), elapsed
