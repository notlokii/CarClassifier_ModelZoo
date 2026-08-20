"""Phase 6 plots: accuracy, time, overlaid losses, confusion matrices.

Confusion matrices use sklearn.metrics (API), not a hand-rolled count.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from common.data import CLASS_NAMES


def plot_train_val_acc(names: list[str], train_acc: list[float], val_acc: list[float]) -> None:
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, train_acc, w, label="train")
    ax.bar(x + w / 2, val_acc, w, label="val")
    ax.axhline(1 / 7, color="gray", linestyle="--", linewidth=1, label="chance (~14%)")
    ax.axhline(0.169, color="tab:orange", linestyle=":", linewidth=1, label="majority (~17%)")
    ax.set_xticks(x, labels=names)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Train vs val accuracy")
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_times(names: list[str], seconds: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, seconds, color="tab:green")
    ax.set_ylabel("seconds")
    ax.set_title("Training wall time")
    fig.tight_layout()
    plt.show()


def plot_loss_overlay(
    numpy_history: dict | None,
    keras_history: dict | None,
    mlp_loss_curve: list[float] | None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(np.log(7), color="gray", linestyle="--", linewidth=1, label="chance (~1.95)")
    if numpy_history is not None:
        ax.plot(numpy_history["train_loss"], label="NumPy train (SGD)")
        ax.plot(numpy_history["val_loss"], label="NumPy val")
    if keras_history is not None:
        ax.plot(keras_history["loss"], label="Keras train (SGD)")
        ax.plot(keras_history["val_loss"], label="Keras val")
    if mlp_loss_curve is not None:
        ax.plot(mlp_loss_curve, label="sklearn MLP train (Adam)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.set_ylim(0, 3.5)
    ax.set_title("Loss curves (y clipped; Keras epoch-1 spike is off-scale)")
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_confusion_grid(y_true: np.ndarray, preds: dict[str, np.ndarray]) -> None:
    n = len(preds)
    ncols = 2 if n > 2 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 5.0 * nrows), layout="constrained"
    )
    axes_flat = np.atleast_1d(axes).ravel()
    labels = list(range(len(CLASS_NAMES)))
    for ax, (name, y_pred) in zip(axes_flat, preds.items()):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
            ax=ax, colorbar=False, xticks_rotation=45
        )
        ax.set_title(name)
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.suptitle("Validation confusion matrices (rows = true class)")
    plt.show()
