# Multiclass Car Classifier — Model Zoo

7-class car body-type classifier (Convertible, Coupe, Hatchback, Pick-Up, SUV, Sedan, VAN)
built three times on the same flattened `64×64×3` images:

1. **`numpy_model/`** — from-scratch NumPy net (forward, backprop, loss, SGD)
2. **`tf_model/`** — Keras twin of that architecture
3. **`sklearn_baseline/`** — logistic regression + Adam MLP

Learning project (Andrew Ng Course 2). Accuracy is not the goal; diagnosing bias/variance
and why flattening has a lower ceiling than a CNN is. Spec: `PRD.md`.

## Layout

```
common/            data pipeline, Kaggle download, comparison plots
numpy_model/       from-scratch net
tf_model/          Keras equivalent
sklearn_baseline/  sklearn baselines
car_classifier_model_zoo.ipynb   lab notebook (run from repo root)
PRD.md
requirements.txt
data/              raw + processed arrays (gitignored)
```

## Setup

```bash
conda activate carzoo          # or mlcourse
pip install -r requirements.txt
python common/download_data.py # once; needs ~/.kaggle/kaggle.json
jupyter notebook car_classifier_model_zoo.ipynb
```

Split is 70/15/15 stratified. Net: `12288 → 128 → 64 → 7`, ReLU hidden, softmax out.

## Results

Val used for diagnosis; test held out. Chance ≈ 14%; majority (Convertible) ≈ 17%.

| Model | Optimizer | Train acc | Val acc | Wall time | Diagnosis |
|---|---|---|---|---|---|
| NumPy MLP | SGD lr=0.5, 20 ep | 0.169 | 0.169 | minutes | high bias |
| Keras MLP | same SGD | 0.166 | 0.169 | 4.7s | high bias (same recipe, faster) |
| Logistic regression | saga, max_iter=200 | 0.982 | 0.500 | 203s | high variance (`p > n`) |
| sklearn MLP | Adam, 20 ep | 0.666 | 0.496 | 17s | learns; some overfit |

**NumPy ≈ Keras** because architecture and SGD lr=0.5 match — both sit on Convertible. The gap is runtime, not math.

**Flattening vs CNN:** a wheel or windshield is a local pattern. Dense layers on 12,288 independent pixels do not share that structure, so ~50% val is a plausible MLP ceiling here. Next step is a CNN (or Adam / a smaller SGD lr in the hand-written net), not regularization on the stuck SGD runs.

Do not shop hyperparameters on test. Score test once after you pick a story.
