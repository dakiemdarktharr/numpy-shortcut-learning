# Reproducing the experiment

## Environment

Tested locally with Python **3.11.9** and NumPy **2.4.6** on Windows. NumPy is the only runtime dependency. Python's standard library handles configuration, CLI, metrics serialization, PNG output, SVG charts and unit tests. Packaging uses setuptools; no ML framework, automatic differentiation or image library is needed.

### Windows PowerShell

```powershell
git clone https://github.com/dakiemdarktharr/numpy-shortcut-learning.git
cd numpy-shortcut-learning
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m shortcut_learning run --config configs/default.json --output runs/default
.\.venv\Scripts\python.exe scripts/verify_run.py runs/default
```

Activation is optional; these commands use the virtual environment's Python directly. The already-created local checkout has its environment and complete run in place.

### Linux / macOS

```sh
git clone https://github.com/dakiemdarktharr/numpy-shortcut-learning.git
cd numpy-shortcut-learning
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m shortcut_learning run --config configs/default.json --output runs/default
.venv/bin/python scripts/verify_run.py runs/default
```

The implementation is portable, but the committed measured snapshot comes from the recorded local environment. Floating-point decisions close to the threshold can differ slightly between numerical backends; inspect the actual counts from your run.

## Run each stage separately

After installation, `shortcut-learning` is also an available entry point. The module form below works without adding the environment's Scripts/bin directory to PATH.

```sh
python -m shortcut_learning generate --output runs/default
python -m shortcut_learning train --output runs/default
python -m shortcut_learning evaluate --output runs/default
python -m shortcut_learning report --output runs/default
```

Use the environment's Python in place of `python` if it is not activated. `run` executes all four stages in order. `evaluate` loads the already-saved weights and never calls training. It checks that dataset hashes match the training metadata and that model files remain byte-for-byte unchanged through evaluation.

Configuration is read from `configs/default.json`. If you change it, use a new `--output` directory. The CLI rejects a different configuration in an existing run directory. Re-running an identical configuration intentionally regenerates that run's artifacts.

To copy a small results snapshot into the repository after a run:

```sh
python -m shortcut_learning snapshot --output runs/default --destination .
```

This copies metrics, example images, charts and model weights. Full dataset archives and the virtual environment stay local and are ignored by Git. The README describes the committed reference snapshot; update its tables when publishing a different configuration.

## Dataset construction and counts

For each seed:

- Generate 1,200 balanced training objects and 1,200 independently generated balanced test objects.
- Render two training views, each with 1,200 rows: biased backgrounds and varied backgrounds. They share masks, labels and noise.
- Render four paired test views, each with 1,200 rows. Both models see the same test inputs in each condition.
- Preserve foreground values exactly between the background interventions. The position diagnostic moves the mask without cropping or wrapping; its background retains the original rule.
- Check that no exact image is shared between any training view and any test view. This check does not remove their shared generator or imply real-world representativeness.

There are **2,400 underlying generated samples per seed**. Counting both training views and all four test views gives **7,200 condition-specific rows per seed**, before accounting for duplicates between views. Do not describe those rows as independent samples. The full five-seed run generates 12,000 underlying samples and saves five dataset archives, approximately 93 MiB in this environment.

Masks are created analytically from a jittered center and radius. Circle pixels satisfy a radius-squared inequality; square pixels satisfy horizontal and vertical bounds. Randomness uses NumPy's generator with explicit seeds. Foreground intensity is 0.02, backgrounds are 0.18/0.82, and additive Gaussian noise has standard deviation 0.012 before clipping to [0,1].

## Inspect the arrays and trained models

```python
from pathlib import Path
import numpy as np
from shortcut_learning.config import Config
from shortcut_learning.data import load_dataset
from shortcut_learning.model import LogisticRegression

config = Config.load("configs/default.json")
data = load_dataset(Path("runs/default/data/seed-101.npz"), 101, config)
print(data["train_biased"].shape)  # (1200, 24, 24)
print(data["test_labels"][:8])    # 0=circle, 1=square

model, metadata = LogisticRegression.load(Path("runs/default/models/seed-101-biased.npz"))
print(model.predict(data["test_background_reversed"][:8]))
print(model.predict_proba(data["test_background_reversed"][:8]))

with np.load("examples/sample-dataset.npz", allow_pickle=False) as sample:
    print(sample.files)
    print(sample["train_biased"].shape)  # (8, 24, 24); this is only a small example archive
```

| Archive key | Shape at reference settings | Meaning |
|---|---|---|
| `train_labels`, `test_labels` | `(1200,)` | Balanced binary class labels |
| `train_masks`, `test_masks`, `shifted_masks` | `(1200,24,24)` | Boolean foreground masks |
| `train_noise`, `test_noise` | `(1200,24,24)` | Shared float32 additive noise |
| `train_biased`, `train_varied` | `(1200,24,24)` | Training inputs for the two models |
| `train_biased_backgrounds`, `train_varied_backgrounds` | `(1200,)` | Training background assignments |
| `test_independent_backgrounds` | `(1200,)` | Background assignments in the independent condition |
| `test_same_rule`, `test_background_reversed` | `(1200,24,24)` | Paired original/reversed background test inputs |
| `test_background_independent`, `test_position_shifted` | `(1200,24,24)` | Remaining test inputs |
| `metadata_json` | scalar string | Dataset seed, configuration and label names |

Model NPZ files contain `weights`, `bias` and `metadata_json`. Loading uses `allow_pickle=False`. Metadata identifies the seed, model, feature transform, threshold and source dataset hash.

## Model and optimization

Flatten each image into 576 features, then subtract 0.5 from every feature. For square as class 1:

```text
z = X @ w + b
p = sigmoid(z)
prediction = 1 if p >= 0.5 else 0

loss = mean(logaddexp(0, z) - y*z) + (l2/2) * sum(w*w)
dw = X.T @ (p-y) / n + l2*w
db = mean(p-y)
w = w - learning_rate*dw
b = b - learning_rate*db
```

The bias is not regularized. Both models start from zero weights and bias; both use 800 steps, learning rate 0.025 and L2 coefficient 0.001. Main training uses float32 features/weights and the same update order for every seed. Loss is recorded before the first step, every 25 steps and after the last step. The sigmoid exponent is clipped to avoid overflow; log loss is calculated from logits with `logaddexp` for stability.

An earlier exploratory design used a midgray foreground and a larger optimizer step; the varied-data model failed to fit its training set. Foreground and step size were adjusted using training-only development runs before freezing the reference configuration and running the five reported seeds. This is not a claim of external preregistration. The position condition was introduced later as an additional diagnostic; this code now reproduces it alongside the original background conditions without retuning.

## Metrics and interpretation

`metrics.json` includes correct/total, accuracy, per-class precision/recall/F1, macro precision/recall/F1, binary log loss, Brier score and confusion matrices. Macro averages give circle and square equal weight. Ratios with a zero denominator use zero. Sigmoid scores have not been calibrated as confidence estimates.

Confusion-matrix rows are actual classes `[circle, square]`, columns are predicted classes `[circle, square]`. `summary.csv` gives mean, population standard deviation, min and max across the five seeds. Those values describe this designed experiment; paired conditions are not independent replications. The paired prediction-flip rate measures how often changing only the background switches the predicted label.

The additional position test preserves the original background-label relationship. Therefore a background-dependent classifier can score perfectly there. Read its result together with the reversed/independent background tests rather than treating it as proof of shape recognition.

## Verification

```sh
python -m unittest discover -s tests -v
python scripts/verify_run.py runs/default --receipt results/verification.json
```

Tests cover central finite differences of the loss gradients, the actual first optimizer update, hand-computed metrics, stable extreme-logit loss, deterministic generation, controlled interventions, no-wrap shifting, invalid inputs, model save/load and an end-to-end small run. The integration test verifies model hashes before and after all evaluations and checks exported artifacts. It also checks that runtime imports are NumPy or standard-library modules.

`verify_run.py` checks the full fixed reference run, not just the small integration fixture: all five seeds, all 50 train/test metric rows, the background-result pattern, the additional position diagnostic, file hashes and source provenance. It is not a validator for arbitrary redesigned datasets.

## Layout

```text
configs/default.json        Fixed reference settings
src/shortcut_learning/     Generator, handwritten model, metrics, stages, reports and CLI
tests/                     Standard-library unittest suite
scripts/verify_run.py       Reference-run and artifact integrity checks
examples/                  Selected PNG examples, small sample NPZ and indices
results/                   Committed measured snapshot, charts and all ten model files
runs/default/              Local full datasets, models and results (Git-ignored)
.venv/                     Isolated local environment (Git-ignored)
```
