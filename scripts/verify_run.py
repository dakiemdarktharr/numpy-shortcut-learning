"""Check the fixed reference protocol's measured results and saved-file integrity."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    run = args.run
    metrics = json.loads((run/"metrics/metrics.json").read_text(encoding="utf-8"))
    require(metrics["config"]["seeds"] == [101, 211, 307, 401, 503], "Expected the five reference seeds")
    rows = metrics["rows"]
    require(len(rows) == 50, "Expected 50 train/test rows")
    for row in rows:
        require(row["n"] == 1200, "Reference rows need n=1200")
        require(sum(map(sum, row["confusion_matrix"])) == 1200, "Confusion counts do not add up")
        require(sum(row["confusion_matrix"][i][i] for i in [0, 1]) == row["correct"], "Correct count differs from confusion")
        require(abs(row["accuracy"]-row["correct"]/row["n"]) < 1e-12, "Accuracy disagrees with counts")
        if row["condition"] == "train" or row["condition"] == "same_rule":
            require(row["correct"] == 1200, "Reference training/same-rule fit failed")
        elif row["condition"] == "background_reversed":
            require(row["correct"] == (0 if row["model"] == "biased" else 1200), "Background reversal pattern changed")
        elif row["condition"] == "background_independent":
            require(row["correct"] == (600 if row["model"] == "biased" else 1200), "Independent-background pattern changed")
        elif row["condition"] == "position_shifted":
            if row["model"] == "biased":
                require(row["correct"] == 1200, "Biased model no longer follows the retained background rule")
            else:
                require(.45 <= row["accuracy"] <= .60, "Position sensitivity no longer matches the reference diagnostic")
    for filename in ["dataset-manifest.json", "model-manifest.json"]:
        manifest = json.loads((run/filename).read_text(encoding="utf-8"))
        for item in manifest.get("datasets", manifest.get("models", [])):
            require(digest(run/item["file"]) == item["sha256"], f"Artifact hash mismatch: {item['file']}")
    require(len(metrics["model_integrity"]) == 10, "Missing evaluation integrity records")
    for item in metrics["model_integrity"]:
        require(item["sha256_before"] == item["sha256_after"] and item["weights_unchanged"], "Weights changed during evaluation")
        expected = 1.0 if item["model"] == "biased" else 0.0
        require(item["paired_background_flip_rate"] == expected, "Unexpected paired prediction-flip rate")
    environment = json.loads((run/"metrics/environment.json").read_text(encoding="utf-8"))
    source = Path(__file__).resolve().parents[1]/"src/shortcut_learning"
    for filename, expected in environment["source_sha256"].items():
        require(digest(source/filename) == expected, f"Source changed since evaluation: {filename}")
    result = {"reference_protocol": "passed", "seeds": 5, "trained_model_files": 10, "metric_rows": 50,
              "dataset_and_model_hashes": "passed", "source_hashes": "passed", "weights_unchanged": True,
              "runtime_dependency": "numpy", "numpy": environment["numpy"], "python": environment["python"]}
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
