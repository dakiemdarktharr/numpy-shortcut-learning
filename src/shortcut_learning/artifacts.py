"""PNG examples and SVG charts using only NumPy and Python's standard library."""
import csv
from html import escape
import json
from pathlib import Path
import shutil
import struct
import zlib
import numpy as np
from shortcut_learning.config import Config
from shortcut_learning.data import CONDITIONS, load_dataset
from shortcut_learning.experiment import MODELS, write_json

__all__ = ["write_png", "report", "snapshot"]
INK, BLUE, RED, PAPER = "#24314d", "#315fb8", "#873c56", "#edf2fa"
LABELS = {"same_rule": ["Same background", "rule"], "background_reversed": ["Reversed", "backgrounds"],
          "background_independent": ["Independent", "backgrounds"], "position_shifted": ["Shifted 6 pixels", "right"]}
MODEL_LABELS = {"biased": "Biased training", "varied": "Varied backgrounds"}


def write_png(path: Path, pixels: np.ndarray, scale: int = 20) -> None:
    """Lossless 8-bit grayscale PNG for illustration; training retains float32 pixels."""
    values = np.asarray(pixels)
    if values.ndim != 2 or not np.isfinite(values).all() or scale < 1:
        raise ValueError("Expected a finite 2-D image and positive scale")
    image = np.rint(np.clip(values, 0, 1)*255).astype(np.uint8)
    image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
    height, width = image.shape

    def chunk(name: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", zlib.crc32(name+payload) & 0xffffffff)

    raw = b"".join(b"\x00"+row.tobytes() for row in image)
    content = b"\x89PNG\r\n\x1a\n"
    content += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    content += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _text(x: float, y: float, value: object, size: int = 23, fill: str = INK, anchor: str = "start") -> str:
    return f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" fill="{fill}" text-anchor="{anchor}">{escape(str(value))}</text>'


def _svg(width: int, height: int, elements: list[str]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            f'<rect width="{width}" height="{height}" fill="{PAPER}"/>' + "".join(elements) + "</svg>\n")


def _accuracy_chart(result: dict, config: Config) -> str:
    rows = {(r["model"], r["condition"]): r for r in result["rows"] if r["seed"] == config.primary_seed}
    elements = [_text(60, 65, "Background shortcut and position sensitivity", 35),
                _text(60, 108, f"Primary seed {config.primary_seed} · {config.n_test:,} held-out images per condition", 23),
                _text(20, 160, "Accuracy (%)", 20)]
    baseline, top = 490, 190
    for value in [0, 25, 50, 75, 100]:
        y = baseline-(baseline-top)*value/100
        elements += [f'<line x1="100" x2="1160" y1="{y}" y2="{y}" stroke="#b9c6dc"/>',
                     _text(85, y+7, value, 21, anchor="end")]
    for j, condition in enumerate(CONDITIONS):
        center = 230+j*265
        for i, name in enumerate(MODELS):
            accuracy = rows[name, condition]["accuracy"]*100
            height = (baseline-top)*accuracy/100
            x = center-77+i*82
            color = RED if name == "biased" else BLUE
            elements.append(f'<rect x="{x}" y="{baseline-height}" width="66" height="{max(height,2)}" fill="{color}"/>')
            label = f"{accuracy:.0f}%" if accuracy.is_integer() else f"{accuracy:.2f}%"
            elements.append(_text(x+33, baseline-height-13, label, 21, anchor="middle"))
        lines = LABELS[condition] if condition != "position_shifted" else [f"Shifted {config.shift_pixels} pixels", "right"]
        for k, line in enumerate(lines):
            elements.append(_text(center, 535+k*29, line, 22, anchor="middle"))
    for x, name, color in [(130, "Biased training", RED), (470, "Varied backgrounds", BLUE)]:
        elements += [f'<rect x="{x}" y="605" width="24" height="24" fill="{color}"/>', _text(x+37, 626, name, 23)]
    elements += [_text(60, 675, "The position condition is an additional diagnostic; weights are unchanged during all tests.", 19)]
    return _svg(1230, 710, elements)


def _loss_chart(output: Path, config: Config) -> str:
    with (output / "metrics/training-history.csv").open(newline="", encoding="utf-8") as file:
        rows = [r for r in csv.DictReader(file) if int(r["seed"]) == config.primary_seed]
    highest = max(float(r["objective"]) for r in rows)*1.12
    elements = [_text(60, 60, "Training objective", 35),
                _text(60, 103, "Mean binary cross entropy + L2 penalty · full-batch gradient descent", 23)]
    for fraction in [0, .25, .5, .75, 1]:
        y = 510-fraction*325
        elements += [f'<line x1="120" x2="1080" y1="{y}" y2="{y}" stroke="#b9c6dc"/>',
                     _text(100, y+7, f"{highest*fraction:.2f}", 21, anchor="end")]
    for name, color in [("biased", RED), ("varied", BLUE)]:
        group = [r for r in rows if r["model"] == name]
        points = " ".join(f'{120+960*int(r["step"])/config.steps},{510-325*float(r["objective"])/highest}' for r in group)
        elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4"/>')
    for step in [0, config.steps//2, config.steps]:
        elements.append(_text(120+960*step/config.steps, 550, step, 22, anchor="middle"))
    elements += [_text(600, 590, "Gradient steps", 24, anchor="middle"),
                 _text(130, 645, "Biased training", 24, RED), _text(470, 645, "Varied backgrounds", 24, BLUE)]
    return _svg(1200, 690, elements)


def _confusion_chart(result: dict, config: Config) -> str:
    elements = [_text(55, 55, "Confusion matrices", 35),
                _text(55, 95, f"Seed {config.primary_seed} · rows = actual, columns = predicted · C = circle, S = square", 22)]
    rows = {(r["model"], r["condition"]): r for r in result["rows"] if r["seed"] == config.primary_seed}
    for i, model in enumerate(MODELS):
        yy = 205+i*295
        elements.append(_text(55, yy-30, MODEL_LABELS[model], 26, RED if model == "biased" else BLUE))
        for j, condition in enumerate(CONDITIONS):
            x, y = 135+j*270, yy+40
            for k, label in enumerate(LABELS[condition]):
                if condition == "position_shifted" and k == 0:
                    label = f"Shifted {config.shift_pixels} pixels"
                elements.append(_text(x+60, yy+k*24, label, 19, anchor="middle"))
            matrix = rows[model, condition]["confusion_matrix"]
            for label in [0, 1]:
                elements.append(_text(x+label*62+30, y-9, ["C", "S"][label], 19, anchor="middle"))
                elements.append(_text(x-14, y+label*62+38, ["C", "S"][label], 19, anchor="middle"))
            for a in [0, 1]:
                for b in [0, 1]:
                    count = matrix[a][b]
                    opacity = .08+.92*count/(config.n_test/2)
                    xx, ycell = x+b*62, y+a*62
                    elements.append(f'<rect x="{xx}" y="{ycell}" width="60" height="60" fill="{BLUE}" fill-opacity="{opacity}"/>')
                    elements.append(_text(xx+30, ycell+39, count, 22, "white" if opacity > .55 else INK, "middle"))
    return _svg(1230, 790, elements)


def _examples(output: Path, config: Config) -> None:
    data = load_dataset(output / "data" / f"seed-{config.primary_seed}.npz", config.primary_seed, config)
    directory = output / "examples"
    records = []
    for model in MODELS:
        backgrounds = data[f"train_{model}_backgrounds"]
        for label, word in [(0, "circle"), (1, "square")]:
            for bg, bg_word in [(0, "dark"), (1, "light")]:
                candidates = np.flatnonzero((data["train_labels"] == label) & (backgrounds == bg))
                if not len(candidates):
                    continue
                index = int(candidates[0])
                filename = f"train-{model}-{word}-{bg_word}.png"
                write_png(directory / filename, data[f"train_{model}"][index])
                records.append({"file": filename, "split": "train", "training_view": model, "index": index,
                                "label": word, "background": bg_word, "selection": "first matching index, not prediction quality"})
    for label, word in [(0, "circle"), (1, "square")]:
        index = int(np.flatnonzero(data["test_labels"] == label)[0])
        for condition in CONDITIONS:
            filename = f"test-{word}-{condition}.png"
            write_png(directory / filename, data[f"test_{condition}"][index])
            records.append({"file": filename, "split": "test", "condition": condition, "index": index,
                            "label": word, "selection": "first matching index, not prediction quality"})
    write_json(directory / "index.json", {"seed": config.primary_seed, "png_status": "8-bit display copies; float32 source used by model",
                                         "examples": records})
    # A small, inspectable sample archive; the complete archives stay in runs/.../data.
    selected = {}
    for split in ["train", "test"]:
        ids = np.sort(np.concatenate([np.flatnonzero(data[f"{split}_labels"] == label)[:4] for label in [0, 1]]))
        for key, values in data.items():
            if key.startswith(split+"_"):
                selected[key] = values[ids]
    np.savez_compressed(directory / "sample-dataset.npz", **selected,
                        sample_metadata_json=np.array(json.dumps({"seed": config.primary_seed, "samples_per_split": 8,
                            "selection": "first four indices in each class", "not_the_full_dataset": True})))


def report(output: Path, config: Config) -> None:
    result = json.loads((output / "metrics/metrics.json").read_text(encoding="utf-8"))
    if result["config"] != config.to_dict():
        raise ValueError("Metrics configuration differs from report configuration")
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    for filename, svg in [("accuracy.svg", _accuracy_chart(result, config)),
                          ("training-loss.svg", _loss_chart(output, config)),
                          ("confusion-matrices.svg", _confusion_chart(result, config))]:
        (figures / filename).write_text(svg, encoding="utf-8")
    _examples(output, config)
    lines = ["# Measured results", "", f"Primary seed: {config.primary_seed}. Counts are per condition; scores are uncalibrated.", "",
             "| Model | Condition | Correct / n | Accuracy | Macro precision | Macro recall | Macro F1 | Log loss |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["rows"]:
        if row["seed"] == config.primary_seed:
            lines.append(f'| {row["model"]} | {row["condition"]} | {row["correct"]}/{row["n"]} | '
                         f'{row["accuracy"]:.4f} | {row["precision_macro"]:.4f} | {row["recall_macro"]:.4f} | '
                         f'{row["f1_macro"]:.4f} | {row["log_loss"]:.4f} |')
    lines += ["", "Full per-class metrics and confusion matrices: metrics.json. All seeds: metrics.csv and summary.csv.",
              "Paired background conditions share source objects and noise. Do not treat them as independent replications.", ""]
    (output / "metrics/summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report: three SVG charts, 14 PNG examples, sample NPZ and metric tables written to {output}", flush=True)


def snapshot(output: Path, destination: Path) -> None:
    """Copy only the small scientific artifacts into a repository, never the environment or full datasets."""
    results = destination / "results"
    results.mkdir(parents=True, exist_ok=True)
    for filename in ["metrics.json", "metrics.csv", "summary.csv", "training-history.csv", "environment.json", "summary.md"]:
        shutil.copy2(output / "metrics" / filename, results / filename)
    for filename in ["config.json", "dataset-manifest.json", "model-manifest.json"]:
        shutil.copy2(output / filename, results / filename)
    for path in (output / "figures").glob("*.svg"):
        shutil.copy2(path, results / path.name)
    shutil.copytree(output / "examples", destination / "examples", dirs_exist_ok=True)
    shutil.copytree(output / "models", results / "models", dirs_exist_ok=True)
    print(f"snapshot: results, trained weights and examples copied to {destination}", flush=True)
