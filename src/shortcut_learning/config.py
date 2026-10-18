"""Explicit, validated configuration; no test-driven parameter search."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

__all__ = ["Config"]


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = (101, 211, 307, 401, 503)
    primary_seed: int = 101
    n_train: int = 1200
    n_test: int = 1200
    image_size: int = 24
    foreground: float = 0.02
    dark_background: float = 0.18
    light_background: float = 0.82
    noise_std: float = 0.012
    center_jitter: float = 0.65
    radius_min: float = 4.55
    radius_max: float = 5.15
    shift_pixels: int = 6
    steps: int = 800
    learning_rate: float = 0.025
    l2: float = 0.001
    history_every: int = 25

    def __post_init__(self) -> None:
        integer_fields = ["primary_seed", "n_train", "n_test", "image_size", "shift_pixels", "steps", "history_every"]
        if any(type(getattr(self, k)) is not int for k in integer_fields):
            raise ValueError("Counts, seeds, image size and shift must be integers")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if any(type(s) is not int or s < 0 for s in self.seeds):
            raise ValueError("seeds must be nonnegative integers")
        if self.primary_seed not in self.seeds:
            raise ValueError("primary_seed must occur in seeds")
        if self.n_train < 4 or self.n_test < 4 or self.n_train % 4 or self.n_test % 4:
            raise ValueError("n_train and n_test must be positive multiples of four")
        if self.image_size < 16 or not 0 <= self.shift_pixels < self.image_size:
            raise ValueError("Invalid image size or horizontal shift")
        floats = [self.foreground, self.dark_background, self.light_background, self.noise_std,
                  self.center_jitter, self.radius_min, self.radius_max, self.learning_rate, self.l2]
        if not all(math.isfinite(v) for v in floats):
            raise ValueError("Numeric configuration values must be finite")
        if not 0 <= self.foreground < self.dark_background < self.light_background <= 1:
            raise ValueError("Require 0 <= foreground < dark background < light background <= 1")
        if self.noise_std < 0 or self.center_jitter < 0 or not 0 < self.radius_min <= self.radius_max:
            raise ValueError("Invalid generator variation")
        if self.steps < 1 or self.history_every < 1 or self.learning_rate <= 0 or self.l2 < 0:
            raise ValueError("Invalid optimizer settings")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["seeds"] = list(self.seeds)
        return result

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if "seeds" in values:
            values["seeds"] = tuple(values["seeds"])
        return cls(**values)
