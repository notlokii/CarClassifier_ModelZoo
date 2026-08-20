"""Phase 5: sklearn baselines on the same flattened arrays.

Logistic regression is the linear sanity check (no hidden layers).
MLPClassifier uses the same hidden sizes as NumPy/Keras (128, 64) but
sklearn's default solver is Adam — not the SGD lr=0.5 from Phases 3–4.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from numpy_model.network import LAYER_DIMS


def _fit(model, X, y):
    """Fit with a fixed iteration budget. Hitting max_iter is expected, not a bug."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, y)
    return model


def train_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
) -> tuple[LogisticRegression, dict, float]:
    """Multinomial logistic regression. Integer labels, not one-hot."""
    model = LogisticRegression(
        solver="saga",
        max_iter=200,
        random_state=seed,
    )
    t0 = time.perf_counter()
    _fit(model, X_train, y_train)
    elapsed = time.perf_counter() - t0
    metrics = {
        "train_acc": float(model.score(X_train, y_train)),
        "val_acc": float(model.score(X_val, y_val)),
    }
    return model, metrics, elapsed


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[MLPClassifier, dict, float]:
    """Same depth/width as LAYER_DIMS; Adam (sklearn default), 20 epochs."""
    hidden = tuple(LAYER_DIMS[1:-1])  # (128, 64)
    model = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        batch_size=batch_size,
        max_iter=epochs,
        random_state=seed,
        shuffle=True,
        verbose=False,
    )
    t0 = time.perf_counter()
    _fit(model, X_train, y_train)
    elapsed = time.perf_counter() - t0
    metrics = {
        "train_acc": float(model.score(X_train, y_train)),
        "val_acc": float(model.score(X_val, y_val)),
        "loss_curve": list(model.loss_curve_),
    }
    return model, metrics, elapsed
