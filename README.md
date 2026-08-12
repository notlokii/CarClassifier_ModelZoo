# Multiclass Car Classifier — Model Zoo Edition

A 7-class car body-type classifier (Convertible, Coupe, Hatchback, Pick-Up, SUV, Sedan, VAN)
built **three times** on the same dataset to compare implementations head-to-head:

1. **`numpy_model/`** — from-scratch NumPy neural net (forward pass, backprop, loss, optimizer all hand-written).
2. **`tf_model/`** — TensorFlow/Keras version with the identical architecture.
3. **`sklearn_baseline/`** — scikit-learn `MLPClassifier` / logistic regression sanity-check baseline.

This is a **learning-first** project (Andrew Ng ML Specialization, Course 2). Maximizing accuracy is an
explicit non-goal — understanding *why* a flattened-pixel feedforward net has a lower ceiling than a CNN,
and diagnosing bias/variance correctly, is the real deliverable. See the PRD for full details.

## Repo layout

```
numpy_model/       # from-scratch implementation
tf_model/          # Keras equivalent
sklearn_baseline/  # baseline model
comparison/        # notebook: accuracy, training time, loss curves, confusion matrices
common/            # shared data pipeline (loading, resizing, flatten, normalize, split)
data/
  raw/             # Kaggle dataset (gitignored)
  processed/       # cached resized/flattened arrays (gitignored)
```

## Setup

```bash
conda activate mlcourse        # Python 3.11
pip install -r requirements.txt
```

## Status

- [ ] Phase 0 — Setup & data pipeline
- [ ] Phase 1 — NumPy forward pass
- [ ] Phase 2 — NumPy backprop
- [ ] Phase 3 — NumPy full training + first diagnosis
- [ ] Phase 4 — TensorFlow version
- [ ] Phase 5 — scikit-learn baseline
- [ ] Phase 6 — Comparison & writeup

## Results

_TBD — accuracy, training time, loss curves, confusion matrices, and the bias/variance +
ceiling-diagnosis writeup (Success Criteria, Section 7 of the PRD)._
