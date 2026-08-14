"""Phase 1: NumPy forward pass only. You fill in the math.

Architecture (PRD baseline — not sacred, we'll revisit after learning curves):

    input  →  hidden 1  →  hidden 2  →  output
    12288  →    128     →     64     →    7
               ReLU          ReLU       softmax

Convention in THIS file (matches common/data.py and Keras):
    rows    = examples (m)
    columns = features / units
    Z = A @ W + b          # not W @ A, which is Andrew Ng's column-example form

Ng's notes use X as (n_x, m). Ours is (m, n_x). Same math, everything transposed.
If a lecture slide shows W of shape (n_l, n_{l-1}), our W is (n_{l-1}, n_l).

Phase 1 done when:
  - a batch comes out as (m, 7)
  - each row of the output sums to 1 (a probability distribution)
  - you can say out loud why every matrix has the shape it has
"""

from __future__ import annotations

import numpy as np

from common.data import FLAT_DIM, N_CLASSES

# [input, hidden1, hidden2, output]
LAYER_DIMS = [FLAT_DIM, 128, 64, N_CLASSES]  # [12288, 128, 64, 7]


def init_params(layer_dims: list[int] = LAYER_DIMS, seed: int = 42) -> dict[str, np.ndarray]:
    """Random weights and zero biases for each layer.

    For layer l (1-indexed), connecting n_in units to n_out units:

        W_l shape: (n_in, n_out)
        b_l shape: (n_out,)          # broadcasts across the m rows of Z

    Example with LAYER_DIMS = [12288, 128, 64, 7]:
        W1 (12288, 128)   b1 (128,)
        W2 (128, 64)      b2 (64,)
        W3 (64, 7)        b3 (7,)

    TODO (you write this): draw W from a small random distribution, set b to zeros.
    Hint: ReLU hidden layers often use He init — scale by sqrt(2 / n_in).
    Hint: if W is too large, softmax will overflow; if it's tiny, ReLU outputs
    are ~0 and the network is "dead" before you even train.
    """
    params = {}
    for l in range(1, len(layer_dims)):
        params[f"W{l}"] = np.random.randn(layer_dims[l-1], layer_dims[l]) * np.sqrt(2 / layer_dims[l-1])
        params[f"b{l}"] = np.zeros(layer_dims[l])
    return params

def relu(Z: np.ndarray) -> np.ndarray:
    """Hidden-layer activation. Elementwise: max(0, z).

    Shape in = shape out. For Z1 of (m, 128), A1 is (m, 128).

    TODO: one NumPy line. No Python for-loops over examples or units.
    """
    return np.maximum(0, Z)


def softmax(Z: np.ndarray) -> np.ndarray:
    """Output activation. Each *row* becomes a probability distribution over 7 classes.

    Input  Z: (m, 7)
    Output A: (m, 7)  with A[i, :].sum() == 1 and A[i, j] >= 0

    Formula for one row:  softmax(z)_j = exp(z_j) / sum_k exp(z_k)

    TODO: vectorize over the whole batch. Softmax is per-example, so the
    sum in the denominator is over axis=1 (the 7 classes), not over m.
    Hint: subtract each row's max before exp — otherwise exp(1000) overflows.
    """
    return np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)



def forward(X: np.ndarray, params: dict[str, np.ndarray]):
    """One batch through the net. No training — just matrix multiplies + activations.

    X:      (m, 12288)     a batch of flattened images
    params: dict from init_params

    Returns
    -------
    A3 : (m, 7)
        softmax probabilities
    cache : dict
        stash Z1, A1, Z2, A2, Z3 (and X if you want) for Phase 2 backprop.
        Forward has to remember the intermediates; backprop cannot recreate
        them from A3 alone.

    Data flow to fill in (say the shapes out loud as you code each line):

        Z1 = ?     # (m, 12288) @ (12288, 128) + (128,)  →  (m, 128)
        A1 = relu(Z1)
        Z2 = ?     # (m, 128)   @ (128, 64)    + (64,)   →  (m, 64)
        A2 = relu(Z2)
        Z3 = ?     # (m, 64)    @ (64, 7)      + (7,)    →  (m, 7)
        A3 = softmax(Z3)

    Vectorize: one matmul per layer for the whole batch. No loop over m.
    """
    cache = {}
    for l in range(1, len(LAYER_DIMS)):
        cache[f"Z{l}"] = X @ params[f"W{l}"] + params[f"b{l}"]
        cache[f"A{l}"] = relu(cache[f"Z{l}"])
        if l == len(LAYER_DIMS) - 1:
            cache[f"A{l}"] = softmax(cache[f"Z{l}"])
        X = cache[f"A{l}"]
    return X, cache
