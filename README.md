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
comparison/        # (optional) extra comparison artifacts; the lab notebook is the main report
common/            # shared data pipeline (loading, resizing, flatten, normalize, split)
car_classifier_model_zoo.ipynb  # one lab notebook, sections per phase
data/
  raw/             # Kaggle dataset (gitignored)
  processed/       # cached resized/flattened arrays (gitignored)
```

## Setup

```bash
conda activate mlcourse        # Python 3.11; `carzoo` also works if that's where you installed deps
pip install -r requirements.txt
```

## Where the code lives

Hybrid on purpose: reusable `.py` modules for anything a model must import; **one** notebook (`car_classifier_model_zoo.ipynb`) as the lab / story, with a section per phase.

| Phase | Where you work |
|---|---|
| 0 — data pipeline | `common/data.py` + Phase 0 section of the notebook |
| 1–3 — NumPy net | `numpy_model/` (you write the math) + matching notebook sections |
| 4 — Keras | `tf_model/` |
| 5 — sklearn baseline | `sklearn_baseline/` |
| 6 — comparison | same notebook, final section |

```bash
jupyter notebook car_classifier_model_zoo.ipynb
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
