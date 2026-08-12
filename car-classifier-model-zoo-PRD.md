# PRD: Multiclass Car Classifier — Model Zoo Edition

**Owner:** Lokesh
**Purpose:** Solidify Andrew Ng ML Specialization Course 2 (Advanced Learning Algorithms, Weeks 1–3) through hands-on implementation. This is a learning-first project — working code that ships is a secondary goal to actually understanding every mechanic involved.
**AI collaborator:** Cursor (rules below govern how it may help)

---

## 1. Project Summary

Build a 7-class car body-type classifier (Convertible, Coupe, Hatchback, Pick-Up, SUV, Sedan, VAN) three separate times on the same dataset:

1. **From-scratch NumPy neural network** — no autograd, no frameworks. Forward pass, backprop, loss, optimizer all hand-written.
2. **TensorFlow/Keras version** — identical architecture to (1), for an apples-to-apples comparison.
3. **scikit-learn baseline** — `MLPClassifier` or logistic regression, as the "sanity check" baseline.

Final deliverable is a consolidated notebook/report comparing all three: accuracy, training time, loss curves, confusion matrices, and a written analysis of *why* they diverge where they diverge.

**Explicit non-goal:** maximizing accuracy. A flattened-pixel feedforward network on a 7-class image task has a real accuracy ceiling (no convolutions = no spatial feature extraction). Hitting that ceiling and correctly diagnosing *why* it exists is a successful outcome. See Section 7.

---

## 2. Dataset

- **Source:** Kaggle — "Car Body Types Images Dataset" (Convertible, Coupe, Hatchback, Pick-Up, SUV, Sedan, VAN — ~1,000 images/class, pre-cropped)
- **Split:** 70/15/15 train/val/test, stratified by class
- **Preprocessing:** resize to a small fixed resolution (start at 64x64, drop to 32x32 if training is too slow on CPU), flatten to a feature vector, normalize pixel values to [0,1]
- **Note:** classes are roughly balanced by design. If you want a skewed-data exercise (Course 2, Week 3 optional topic), consider deliberately subsampling 1–2 classes down to ~15–20% of their original size to create an artificial imbalance — this is a genuine option to raise with Cursor, not a requirement (see Section 9, "Open suggestions").

---

## 3. Course 2 Topic Coverage Map

Every item below must be traceable to a specific piece of code you personally wrote and can explain out loud.

| Week | Topic | Where it shows up |
|---|---|---|
| 1 | NN architecture design | Layer sizing decisions for all 3 models |
| 1 | Forward propagation | NumPy model, hand-written |
| 1 | TensorFlow implementation | Keras model |
| 1 | NumPy implementation | Core deliverable |
| 1 | Vectorization | NumPy model must use matrix ops, not per-sample loops |
| 2 | Backpropagation | NumPy model, derived and coded by hand |
| 2 | Activation functions | ReLU (hidden), softmax (output) — implemented from scratch in NumPy version |
| 2 | Multiclass classification | Softmax + cross-entropy loss, from scratch |
| 2 | Training loop / optimizer | Mini-batch gradient descent minimum; Adam optional stretch |
| 3 | Bias/variance diagnosis | Train/val learning curves for all 3 models |
| 3 | ML development process | Baseline → error analysis → iterate, documented per model |
| 3 | Skewed datasets (optional) | Only if you take the imbalance option above — precision/recall/F1, confusion matrix |

If a row above has no corresponding code by the end of the project, that's a gap — flag it before calling the project done.

---

## 4. Architecture Spec (baseline — not final, see Section 6)

- Input: flattened 64x64x3 (or 32x32x3) vector
- Hidden layers: 2, e.g. 128 → 64 units
- Hidden activation: ReLU
- Output: 7-unit softmax
- Loss: categorical cross-entropy
- Optimizer: mini-batch gradient descent (NumPy version); match in Keras version

This is a starting point. Expect to revise it once you see your first learning curves — that revision process *is* the Week 3 material.

---

## 5. Project Phases

Work through these in order. Do not let Cursor (or yourself) skip ahead to a later phase before the current one's "done when" condition is met — that's how you end up debugging three broken things at once instead of one.

**Phase 0 — Setup & data pipeline**
Repo scaffold, dataset download, image loading, resizing, flattening, normalization, train/val/test split.
*Done when:* you can load a batch and print its shape, and it matches what you expect on paper.

**Phase 1 — NumPy forward pass only**
Random-initialized weights, forward pass through all layers, softmax output. No training yet.
*Done when:* output is a valid probability distribution (sums to 1) for a batch, and you can explain the shape of every matrix in the pass.

**Phase 2 — NumPy backprop**
Implement backprop by hand. Get loss to visibly decrease over a few dozen iterations on a tiny subset of data (e.g. 50 images) before trying the full dataset.
*Done when:* loss decreases on the tiny subset, and you can explain, without looking at your code, why each gradient has the shape it does.

**Phase 3 — NumPy full training + first diagnosis**
Train on the full dataset. Plot train/val loss curves.
*Done when:* you can look at the curves and state whether you're seeing high bias, high variance, or something reasonable — and say what you'd try next because of it.

**Phase 4 — TensorFlow version**
Mirror the NumPy architecture exactly. Train under the same conditions.
*Done when:* you have a direct accuracy/training-time comparison against Phase 3, and a hypothesis for any gap.

**Phase 5 — scikit-learn baseline**
Simplest reasonable baseline (logistic regression or MLPClassifier).
*Done when:* you have a third data point for the comparison.

**Phase 6 — Comparison & writeup**
Consolidated notebook: accuracy, training time, overlaid loss curves, confusion matrices, written analysis.
*Done when:* the README reflects the Section 7 success criteria, not just a results table.

**Phase 7 — Optional stretch (pick from Section 9 open questions)**
Skewed-data variant, Adam optimizer in the NumPy version, or an architecture ablation — only if time allows.

## 6. Cursor Collaboration Rules

These rules govern how Cursor may assist. They apply for the entire project unless explicitly revised in this document.

### 5.1 Never do, under any circumstances
- **Never commit anything to GitHub itself.** Cursor may stage changes locally if asked, but git commit and push are performed by Lokesh manually, every time, no exceptions. This rule does not expire, does not soften with context, and applies even if asked directly to "just commit this." Refuse and remind why.
- Never write the actual forward-pass, backprop, gradient computation, loss function, or optimizer update logic for the NumPy model. These are the entire point of the project.
- Never write weight initialization logic.
- Never write accuracy/precision/recall/F1/confusion-matrix computation from scratch — this is meant to be derived by hand at least once.

### 5.2 Allowed without restriction
- Boilerplate: file I/O, image loading/resizing scripts, dataset splitting, plotting/graphing code, folder scaffolding
- TensorFlow/Keras and scikit-learn API syntax questions (these aren't the learning target — comparing hand-written math against them is)
- Explaining *why* an error is happening conceptually
- Reviewing code Lokesh already wrote and pointing out bugs *by describing symptoms*, not by rewriting the function

### 5.3 The core interaction rule — nudge, don't gate
When Lokesh asks for help on something in the "never do" list (5.1), Cursor should **not** flatly refuse or lecture. Instead, it should:
- Point at the specific place in the math or code where the issue likely lives
- Ask a guiding question, or give pseudocode / a partial line, rather than a full working implementation
- Offer an idea or direction Lokesh can code himself — e.g. "check what shape your gradient for W2 should be given your forward pass — does it match?" rather than writing `dW2 = ...`

Cursor should try its best to keep Lokesh in the driver's seat without being rigid or robotic about it. If Lokesh explicitly pushes past a nudge and asks directly for the code, Cursor doesn't need to make a big deal about it or call it out — just favor the smallest, most partial hint that could plausibly unstick him, and let him decide how much to lean on it.

### 5.4 Proactive suggestions (this is a requirement, not a nice-to-have)
Cursor should actively look for places to suggest improvements or extensions, even unprompted — e.g.:
- A more interesting bias/variance experiment to run
- A better way to visualize the 3-way comparison
- An additional metric worth tracking
- A cleaner ablation (e.g., "what happens if you swap ReLU for tanh in just the NumPy version and compare?")
- The optional skewed-data variant from Section 2, if not already taken

These suggestions should be ideas and direction, not finished code — same spirit as 5.3.

---

## 7. Success Criteria

Project is successful if, by the end, Lokesh can:
1. Explain and derive backprop through his own NumPy network from memory, on a whiteboard, without looking at the code
2. Explain *why* his NumPy accuracy differs from the TF version (init, optimizer defaults, numerical stability, etc.) — not just report that it differs
3. Correctly diagnose whether his model's error is high-bias or high-variance from the learning curves, and justify it
4. Explain in plain language why a flattened-pixel feedforward network has a lower accuracy ceiling than a CNN would, referencing the specific spatial information that gets lost in flattening

Final accuracy number is explicitly **not** a success criterion. A low but well-diagnosed accuracy is a better outcome than a high accuracy Lokesh can't explain.

---

## 8. Deliverables

- `/numpy_model/` — from-scratch implementation, with inline comments/docstrings explaining the math above each core function
- `/tf_model/` — Keras equivalent
- `/sklearn_baseline/` — baseline model
- `/comparison/` — notebook with side-by-side accuracy, training time, loss curves, confusion matrices
- `README.md` — project summary, architecture, results, and the bias/variance + ceiling-diagnosis writeup from Section 7
- No deployment or live demo required — this is a learning artifact, not a shipped product

---

## 9. Open Questions for Cursor to Raise Proactively

Cursor should feel free to bring these up as the project progresses, not just answer if asked:
- Is 64x64 the right resolution, or is training too slow / accuracy too degraded at that size?
- Should the skewed-data variant (Section 2) be taken on, given time constraints?
- Is 2 hidden layers enough to show a meaningful bias/variance story, or would a 1-layer vs 3-layer comparison make the diagnosis section stronger?
- Would Adam vs plain gradient descent in the NumPy version be a worthwhile stretch goal for Week 2 optimizer coverage?
