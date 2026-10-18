"""Reproducible generate/train/evaluate stages with persisted data and models."""
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
from shortcut_learning.config import Config
from shortcut_learning.data import CONDITIONS, array_hash, load_dataset, make_dataset, save_dataset
from shortcut_learning.metrics import classification_metrics
from shortcut_learning.model import LogisticRegression

__all__ = ["MODELS", "file_hash", "write_json", "prepare", "generate", "train", "evaluate"]
MODELS = ("biased", "varied")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(output: Path, config: Config) -> None:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "config.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != config.to_dict():
        raise ValueError("Output already belongs to a different configuration; choose another --output")
    write_json(path, config.to_dict())


def generate(output: Path, config: Config) -> dict:
    prepare(output, config)
    manifest = {"base_samples_per_seed": config.n_train + config.n_test,
                "training_rows_per_model": config.n_train, "test_rows_per_condition": config.n_test,
                "paired_conditions": True, "dataset_source": "generated locally; no external images", "datasets": []}
    for seed in config.seeds:
        data = make_dataset(seed, config)
        path = output / "data" / f"seed-{seed}.npz"
        save_dataset(path, data, seed, config)
        manifest["datasets"].append({"seed": seed, "file": f"data/{path.name}", "sha256": file_hash(path),
                                     "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": array_hash(v)}
                                                for k, v in data.items()}})
        print(f"generate seed={seed}: train={config.n_train}, test={config.n_test}, four paired test conditions", flush=True)
    write_json(output / "dataset-manifest.json", manifest)
    return manifest


def train(output: Path, config: Config) -> dict:
    prepare(output, config)
    history, records = [], []
    for seed in config.seeds:
        dataset_path = output / "data" / f"seed-{seed}.npz"
        data = load_dataset(dataset_path, seed, config)
        digest = file_hash(dataset_path)
        for name in MODELS:
            model = LogisticRegression(config.steps, config.learning_rate, config.l2, config.history_every)
            model.fit(data[f"train_{name}"], data["train_labels"])
            path = output / "models" / f"seed-{seed}-{name}.npz"
            metadata = {"seed": seed, "model": name, "config": config.to_dict(), "dataset_sha256": digest,
                        "feature_transform": "flatten raw grayscale pixels, then subtract 0.5",
                        "label_names": ["circle", "square"], "threshold": 0.5, "score_status": "uncalibrated"}
            model.save(path, metadata)
            for row in model.history:
                history.append({"seed": seed, "model": name, **row})
            records.append({"seed": seed, "model": name, "file": f"models/{path.name}", "sha256": file_hash(path)})
            print(f"train seed={seed} model={name}: objective {model.history[0]['objective']:.6f} -> {model.history[-1]['objective']:.6f}", flush=True)
    metrics = output / "metrics"
    metrics.mkdir(exist_ok=True)
    with (metrics / "training-history.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["seed", "model", "step", "data_loss", "objective"])
        writer.writeheader(); writer.writerows(history)
    result = {"optimizer": "handwritten full-batch gradient descent", "models": records}
    write_json(output / "model-manifest.json", result)
    return result


def evaluate(output: Path, config: Config) -> dict:
    prepare(output, config)
    rows, model_checks = [], []
    for seed in config.seeds:
        dataset_path = output / "data" / f"seed-{seed}.npz"
        data = load_dataset(dataset_path, seed, config)
        dataset_digest = file_hash(dataset_path)
        for name in MODELS:
            path = output / "models" / f"seed-{seed}-{name}.npz"
            before = file_hash(path)
            model, metadata = LogisticRegression.load(path)
            if metadata["seed"] != seed or metadata["model"] != name or metadata["config"] != config.to_dict():
                raise ValueError("Model configuration does not match this experiment")
            if metadata["dataset_sha256"] != dataset_digest:
                raise ValueError("Dataset changed after training")
            for condition in ("train",) + CONDITIONS:
                images = data[f"train_{name}"] if condition == "train" else data[f"test_{condition}"]
                labels = data["train_labels"] if condition == "train" else data["test_labels"]
                logits = model.decision_function(images)
                scores = model.predict_proba(images)
                stats = classification_metrics(labels, scores, logits)
                rows.append({"seed": seed, "model": name, "condition": condition, **stats})
            flip_rate = float(np.mean(model.predict(data["test_same_rule"]) != model.predict(data["test_background_reversed"])))
            after = file_hash(path)
            if before != after:
                raise RuntimeError("Evaluation changed a model artifact")
            model_checks.append({"seed": seed, "model": name, "sha256_before": before,
                                 "sha256_after": after, "weights_unchanged": True,
                                 "paired_background_flip_rate": flip_rate})
            test_rows = [r for r in rows if r["seed"] == seed and r["model"] == name and r["condition"] != "train"]
            print(f"evaluate seed={seed} model={name}: " + ", ".join(f"{r['condition']}={r['correct']}/{r['n']}" for r in test_rows), flush=True)
    numeric = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "log_loss", "brier_score"]
    summary = []
    for name in MODELS:
        for condition in ("train",) + CONDITIONS:
            group = [r for r in rows if r["model"] == name and r["condition"] == condition]
            record = {"model": name, "condition": condition, "seeds": len(group), "n_per_seed": group[0]["n"]}
            for key in numeric:
                values = np.array([r[key] for r in group])
                for suffix, value in [("mean", values.mean()), ("std", values.std(ddof=0)),
                                      ("min", values.min()), ("max", values.max())]:
                    record[f"{key}_{suffix}"] = float(value)
            summary.append(record)
    result = {"experiment": "Synthetic circle/square background shortcut", "config": config.to_dict(),
              "model": "L2 logistic regression on raw pixels; handwritten NumPy gradients",
              "conditions": {"same_rule": "circle dark, square light",
                             "background_reversed": "circle light, square dark; same masks, labels and noise",
                             "background_independent": "background balanced within each shape class",
                             "position_shifted": "same-rule background; object mask shifted right without wrapping"},
              "position_test_status": "Additional diagnostic introduced after the background experiment; no retuning",
              "evaluation_retrains_models": False, "paired_conditions_are_independent": False,
              "model_integrity": model_checks, "rows": rows, "summary": summary}
    metrics = output / "metrics"
    metrics.mkdir(exist_ok=True)
    write_json(metrics / "metrics.json", result)
    with (metrics / "metrics.csv").open("w", newline="", encoding="utf-8") as target:
        fields = ["seed", "model", "condition", "correct", "n", *numeric]
        writer = csv.DictWriter(target, fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    with (metrics / "summary.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    source_root = Path(__file__).parent
    environment = {"python": platform.python_version(), "numpy": np.__version__, "platform": sys.platform,
                   "runtime_dependencies": ["numpy"], "parameter_dtype": "float32",
                   "source_sha256": {p.name: file_hash(p) for p in sorted(source_root.glob("*.py"))}}
    write_json(metrics / "environment.json", environment)
    return result
