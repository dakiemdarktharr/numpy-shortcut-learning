"""CLI stages: generate, train, evaluate, report, run and snapshot."""
import argparse
import os
from pathlib import Path

# Set before importing NumPy; explicit user settings remain respected.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from shortcut_learning.config import Config
from shortcut_learning.experiment import evaluate, generate, train

__all__ = ["main"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["generate", "train", "evaluate", "report", "run", "snapshot"])
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/default"))
    parser.add_argument("--destination", type=Path, default=Path("."), help="Repository root for the snapshot stage")
    args = parser.parse_args(argv)
    try:
        config = Config.load(args.config)
        if args.stage in {"generate", "run"}:
            generate(args.output, config)
        if args.stage in {"train", "run"}:
            train(args.output, config)
        if args.stage in {"evaluate", "run"}:
            evaluate(args.output, config)
        if args.stage in {"report", "run"}:
            from shortcut_learning.artifacts import report
            report(args.output, config)
        if args.stage == "snapshot":
            from shortcut_learning.artifacts import snapshot
            snapshot(args.output, args.destination)
    except (ValueError, FileNotFoundError, FloatingPointError) as error:
        parser.exit(2, f"error: {error}\n")
