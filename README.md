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
comparison/        # Phase 6 plot helpers (accuracy, time, loss overlay, confusion matrices)
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
| 6 — comparison | `comparison/plots.py` + final notebook section |

```bash
jupyter notebook car_classifier_model_zoo.ipynb
```

Run from the **repo root**. Same split for every model: 70/15/15 stratified, input `64×64×3` flattened to 12,288 features, 7 classes. Architecture for the nets: `12288 → 128 → 64 → 7`, ReLU hidden, softmax out.

## Status

- [x] Phase 0 — Setup & data pipeline
- [x] Phase 1 — NumPy forward pass
- [x] Phase 2 — NumPy backprop
- [x] Phase 3 — NumPy full training + first diagnosis
- [x] Phase 4 — TensorFlow version
- [x] Phase 5 — scikit-learn baseline
- [x] Phase 6 — Comparison & writeup

## Results

Numbers from the lab notebook (val used for diagnosis; test still held out). Chance ≈ 14%; majority class (Convertible) ≈ 17%.

| Model | Optimizer | Train acc | Val acc | Wall time | Bias / variance |
|---|---|---|---|---|---|
| NumPy MLP | SGD lr=0.5, 20 epochs | 0.169 | 0.169 | minutes (not timed) | **High bias** — majority class |
| Keras MLP (same net) | SGD lr=0.5, 20 epochs | 0.166 | 0.169 | **4.7s** | **High bias** — same recipe, faster |
| Logistic regression | saga, max_iter=200 | 0.982 | 0.500 | 202.9s | **High variance** — 12,288 features vs 5,283 rows |
| sklearn MLP | Adam, 20 epochs | 0.666 | 0.496 | 16.8s | Learns, then some overfit; not fully converged |

Plots live in the Phase 6 notebook section: grouped train/val accuracy, training time, overlaid losses, and val confusion matrices (`sklearn.metrics`).

### Why NumPy and Keras match (and why that is good)

They share architecture, loss, mini-batch size, and **SGD at lr=0.5**. Both collapse to ~17% and (on val) almost always pick Convertible. Keras epoch 1 train loss spiked (~1286) then sat at chance (~1.94), same as NumPy’s flat ~1.94 curve. The gap is **wall time**, not intelligence: Keras runs compiled kernels; NumPy is your loops plus BLAS. Agreement means the hand-written forward/backprop is in the same failure mode as the framework twin — the *training recipe* is the bottleneck, not a silent shape bug unique to NumPy.

### Bias vs variance

- **SGD nets:** train ≈ val ≈ majority. That is high **bias**. Regularization would make it worse. A smaller learning rate or Adam is the first lever (sklearn’s Adam MLP already shows the net can move).
- **Logistic regression:** train 98% vs val 50%. That is high **variance**. A linear model can memorize when features outnumber examples. Stronger L2 (smaller `C`), fewer features, or more data — not a deeper MLP — is the Course 2 playbook.
- **sklearn MLP + Adam:** train 67% / val 50%. Hidden layers help *fit*, but val lands next to logreg. Extra capacity on **flattened** pixels did not raise the generalization ceiling here.

### Flattened pixels vs a CNN

Each image is a 64×64×3 grid. Flattening to 12,288 independent numbers throws away neighborhood structure: a wheel, a windshield, or a pickup bed is a *local* pattern. A dense layer has to relearn “these pixels were neighbors” from weights alone. That is why ~50% val is a plausible ceiling for this MLP zoo, and why a CNN is the honest next architecture — convolutions share filters across space instead of treating every pixel as an unrelated feature.

### What each model is for, and what to improve next

- **NumPy:** whiteboard backprop. Keep it as the teaching artifact. Next: Adam or much smaller SGD lr (optional Phase 7) so the hand-written net can leave chance.
- **Keras:** the iteration engine (and later a `Conv2D` stack) with the same split and class names.
- **Logreg:** linear sanity check and variance demo. Next: stronger regularization if you stay linear.
- **sklearn Adam MLP:** proof that optimizer choice mattered more than “neural nets don’t work.” Next: more epochs or a CNN, not dropout on the stuck SGD runs.

Do not use the test set to shop hyperparameters. When you commit to one story, score test **once**.
