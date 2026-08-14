"""Shared data pipeline: load, resize, flatten, normalize, split.

All three models (NumPy, Keras, sklearn) should import from here so they
train on identical arrays.

Shapes to expect at the default 64x64 RGB setting:

    one image, unflattened : (64, 64, 3)
    one image, flattened   : (12288,)          # 64 * 64 * 3
    a batch of m images    : (m, 12288)
    labels                 : (m,)              # ints in {0, ..., 6}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "Cars_Body_Type"
PROCESSED_DIR = ROOT / "data" / "processed"

CLASS_NAMES = (
    "Convertible",
    "Coupe",
    "Hatchback",
    "Pick-Up",
    "SUV",
    "Sedan",
    "VAN",
)
IMAGE_SIZE = 64  # start here; drop to 32 if CPU training is too slow
N_CLASSES = len(CLASS_NAMES)
FLAT_DIM = IMAGE_SIZE * IMAGE_SIZE * 3  # 12288 at 64x64x3
SPLITS = ("train", "valid", "test")


def _class_to_index() -> dict[str, int]:
    return {name: i for i, name in enumerate(CLASS_NAMES)}


def list_image_paths() -> tuple[list[Path], np.ndarray]:
    """Walk data/raw and return (paths, integer labels).

    Every jpg under train/valid/test is pooled so we can re-split 70/15/15.
    To keep Kaggle's original folders, call load_provided_splits() instead.
    """
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw data not found at {RAW_DIR}. Run: python common/download_data.py"
        )

    class_index = _class_to_index()
    paths: list[Path] = []
    labels: list[int] = []

    for split in SPLITS:
        split_dir = RAW_DIR / split
        if not split_dir.is_dir():
            continue
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.glob("*.jpg")):
                paths.append(path)
                labels.append(class_index[class_name])

    if not paths:
        raise FileNotFoundError(f"No .jpg files found under {RAW_DIR}")

    return paths, np.array(labels, dtype=np.int64)


def _resize_one(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        rgb = img.convert("RGB").resize(
            (image_size, image_size), Image.Resampling.LANCZOS
        )
        return np.asarray(rgb, dtype=np.uint8)


def load_resized_images(
    image_size: int = IMAGE_SIZE,
    cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load every image, resize, return X (m, H, W, 3) uint8 and y (m,) int.

    Resized arrays are cached under data/processed/ so we don't redo 7k
    resizes on every run. Delete the .npz (or pass cache=False) to rebuild.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = PROCESSED_DIR / f"resized_{image_size}x{image_size}.npz"

    if cache and cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        return data["X"], data["y"]

    paths, y = list_image_paths()
    X = np.empty((len(paths), image_size, image_size, 3), dtype=np.uint8)
    for i, path in enumerate(tqdm(paths, desc=f"Resizing to {image_size}x{image_size}")):
        X[i] = _resize_one(path, image_size)

    if cache:
        np.savez_compressed(cache_path, X=X, y=y)

    return X, y


def preprocess(
    X: np.ndarray,
    flatten: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """uint8 (m, H, W, 3) -> float32, optionally flattened to (m, H*W*3)."""
    out = X.astype(np.float32)
    if normalize:
        out = out / 255.0
    if flatten:
        out = out.reshape(out.shape[0], -1)
    return out


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Two-step stratified split. Fractions must sum to 1."""
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_frac,
        stratify=y,
        random_state=random_state,
    )
    # val_frac of the *original* set is val_frac / (1 - test_frac) of what's left
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_frac / (train_frac + val_frac),
        stratify=y_trainval,
        random_state=random_state,
    )
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def load_provided_splits(
    image_size: int = IMAGE_SIZE,
    flatten: bool = True,
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    """Keep Kaggle/Roboflow's existing train/valid/test folders instead of re-splitting."""
    class_index = _class_to_index()
    out: dict[str, np.ndarray] = {}
    folder_to_key = {"train": "train", "valid": "val", "test": "test"}

    for folder, key in folder_to_key.items():
        paths: list[Path] = []
        labels: list[int] = []
        for class_name in CLASS_NAMES:
            class_dir = RAW_DIR / folder / class_name
            for path in sorted(class_dir.glob("*.jpg")):
                paths.append(path)
                labels.append(class_index[class_name])
        X = np.empty((len(paths), image_size, image_size, 3), dtype=np.uint8)
        for i, path in enumerate(tqdm(paths, desc=f"Resizing {folder}")):
            X[i] = _resize_one(path, image_size)
        out[f"X_{key}"] = preprocess(X, flatten=flatten, normalize=normalize)
        out[f"y_{key}"] = np.array(labels, dtype=np.int64)

    out["class_names"] = CLASS_NAMES
    return out


def load_dataset(
    image_size: int = IMAGE_SIZE,
    flatten: bool = True,
    normalize: bool = True,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
    cache: bool = True,
    use_provided_splits: bool = False,
) -> dict[str, np.ndarray]:
    """End-to-end pipeline used by every model.

    Returns a dict with X_train/y_train, X_val/y_val, X_test/y_test, class_names.
    Default: pool all images and stratified-split 70/15/15 (PRD Section 2).
    """
    if use_provided_splits:
        return load_provided_splits(
            image_size=image_size, flatten=flatten, normalize=normalize
        )

    X_hwc, y = load_resized_images(image_size=image_size, cache=cache)
    X = preprocess(X_hwc, flatten=flatten, normalize=normalize)
    data = stratified_split(
        X,
        y,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        random_state=random_state,
    )
    data["class_names"] = CLASS_NAMES
    return data


def to_one_hot(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """Integer labels (m,) -> one-hot (m, n_classes). Handy for the NumPy model later."""
    return np.eye(n_classes, dtype=np.float32)[y]


def load_batch(X: np.ndarray, y: np.ndarray, batch_size: int = 32, start: int = 0):
    """Slice a contiguous mini-batch. Phase 0 sanity check, not a full DataLoader."""
    end = start + batch_size
    return X[start:end], y[start:end]


if __name__ == "__main__":
    data = load_dataset()
    print("class_names:", list(data["class_names"]))
    for split in ("train", "val", "test"):
        X, y = data[f"X_{split}"], data[f"y_{split}"]
        print(f"{split:5}  X {X.shape}  y {y.shape}  X.dtype={X.dtype}  range=[{X.min():.3f}, {X.max():.3f}]")
    Xb, yb = load_batch(data["X_train"], data["y_train"], batch_size=32)
    print(f"batch  X {Xb.shape}  y {yb.shape}")
    print(f"expected flat dim: {FLAT_DIM}  (image_size={IMAGE_SIZE})")
