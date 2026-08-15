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

Phase 2: loss + backprop + a vanilla GD step, proven on ~50 images first.
Phase 3: mini-batch GD on the full train set; diagnose train vs val curves.
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
    np.random.seed(seed)
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
    Z = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=1, keepdims=True)



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
    # Phase 2 needs the original images for dW1. This loop overwrites X.
    # Stash a copy in cache before the first iteration.
    cache["X"] = X
    for l in range(1, len(LAYER_DIMS)):
        cache[f"Z{l}"] = X @ params[f"W{l}"] + params[f"b{l}"]
        cache[f"A{l}"] = relu(cache[f"Z{l}"])
        if l == len(LAYER_DIMS) - 1:
            cache[f"A{l}"] = softmax(cache[f"Z{l}"])
        X = cache[f"A{l}"]
    return X, cache


def relu_grad(Z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 where Z > 0, else 0. Same shape as Z.

    Backprop uses this elementwise:  dZ = dA * relu_grad(Z)

    TODO: one NumPy comparison, no loops. What dtype should the result be so
    it can multiply a float gradient?
    """
    return (Z > 0).astype(Z.dtype)


def compute_loss(A3: np.ndarray, Y: np.ndarray) -> float:
    """Mean categorical cross-entropy.

    A3: (m, 7) softmax probabilities
    Y:  (m, 7) one-hot labels  (see common.data.to_one_hot)

    L = - (1/m) * sum over examples and classes of  Y * log(A3)

    TODO: vectorize. log(0) is -inf — clip A3 with a tiny epsilon first.
    Return a Python float (or a 0-d numpy scalar).
    """
    return -np.sum(Y * np.log(A3 + 1e-10)) / A3.shape[0]



def backward(
    A3: np.ndarray,
    Y: np.ndarray,
    cache: dict[str, np.ndarray],
    params: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Gradients of the mean CCE loss w.r.t. every W and b.

    Returns a dict with the same keys as params: dW1, db1, dW2, db2, dW3, db3.
    Each dW / db must have the *same shape* as the matching W / b.

    Walk backwards. Course 2 fact: softmax + categorical cross-entropy collapse
    into a simple dZ3 in terms of A3 and Y. (Do you divide by m? Your loss is
    a mean — the gradient must match that, or the learning rate will be lying.)

    Then, for a linear layer Z = A_prev @ W + b, given dZ:
      - dW must be (n_in, n_out). Which two matrices multiply to that?
      - db must be (n_out,). You are summing over the batch axis.
      - dA_prev must be (m, n_in), so the previous layer can continue.
      - If that previous layer was ReLU: dZ_prev = dA_prev * relu_grad(Z_prev)

    Shapes to hit (m = batch size):

        dZ3 (m, 7)      dW3 (64, 7)     db3 (7,)
        dZ2 (m, 64)     dW2 (128, 64)   db2 (64,)
        dZ1 (m, 128)    dW1 (12288, 128) db1 (128,)

    Vectorize over m. A loop over the 3 layers is fine.
    You need the original X out of cache for dW1 — see the note in forward.
    """
    m = A3.shape[0]
    # Mean CCE + softmax: dL/dZ3 = (A3 - Y) / m
    dZ3 = (A3 - Y) / m
    dW3 = cache["A2"].T @ dZ3
    db3 = np.sum(dZ3, axis=0)

    dA2 = dZ3 @ params["W3"].T
    dZ2 = dA2 * relu_grad(cache["Z2"])
    dW2 = cache["A1"].T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * relu_grad(cache["Z1"])
    dW1 = cache["X"].T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}


def update_params(
    params: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    learning_rate: float,
) -> dict[str, np.ndarray]:
    """Vanilla gradient descent: step each W and b opposite its gradient.

    TODO: for every key, new = old - learning_rate * grad.
    Same keys as params. In-place or a new dict, your choice.
    """
    for key in params:
        params[key] = params[key] - learning_rate * grads[f"d{key}"]
    return params


def accuracy(A3: np.ndarray, y: np.ndarray) -> float:
    """Fraction of examples whose predicted class matches y.

    A3: (m, 7) softmax probabilities
    y:  (m,) integer labels in {0, ..., 6}

    Predicted class = argmax over the 7 columns (axis=1). Then compare to y
    and take the mean. No sklearn.metrics — this is the one you derive once.
    """
    predicted_classes = np.argmax(A3, axis=1)
    return np.mean(predicted_classes == y) 


def minibatches(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: int | None = None,
):
    """Yield (X_batch, Y_batch) covering every example once.

    Mini-batch GD: each update uses a slice of the data, not all m at once.
    Shuffle every epoch so the batches are not the same ordering every time.
    """
    m = X.shape[0]
    indices = np.arange(m)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, m, batch_size):
        idx = indices[start : start + batch_size]
        yield X[idx], Y[idx]


def train(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.5,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    """Mini-batch gradient descent on the full train set.

    One epoch = one pass through every training example, in shuffled batches.
    After each epoch we record:
      train_loss — mean CCE over the batches that epoch
      val_loss   — CCE on the whole val set (forward only, no update)

    Val is for diagnosis (bias vs variance). Do not pick hyperparameters on test.
    """
    params = init_params(seed=seed)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        batch_losses = []
        for Xb, Yb in minibatches(
            X_train, Y_train, batch_size=batch_size, shuffle=True, seed=seed + epoch
        ):
            A3, cache = forward(Xb, params)
            loss = compute_loss(A3, Yb)
            grads = backward(A3, Yb, cache, params)
            params = update_params(params, grads, learning_rate)
            batch_losses.append(float(loss))

        train_loss = float(np.mean(batch_losses))
        A_val, _ = forward(X_val, params)
        val_loss = float(compute_loss(A_val, Y_val))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch + 1:3d}/{epochs}  train {train_loss:.4f}  val {val_loss:.4f}")

    return params, history

