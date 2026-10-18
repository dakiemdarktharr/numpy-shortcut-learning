"""Synthetic paired datasets, with no download or external annotation dependency."""
import hashlib
import json
from pathlib import Path
import numpy as np
from shortcut_learning.config import Config

__all__ = ["CONDITIONS", "generate_objects", "balanced_backgrounds", "render", "shift_right",
           "make_dataset", "validate_dataset", "save_dataset", "load_dataset", "array_hash"]

CONDITIONS = ("same_rule", "background_reversed", "background_independent", "position_shifted")


def generate_objects(seed: int, n: int, config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return labels (0=circle, 1=square), foreground masks and additive noise."""
    if n < 4 or n % 4:
        raise ValueError("Object count must be a positive multiple of four")
    rng = np.random.default_rng(seed)
    labels = np.tile(np.array([0, 1], dtype=np.int64), n // 2)
    rng.shuffle(labels)
    yy, xx = np.mgrid[:config.image_size, :config.image_size]
    masks = []
    for label in labels:
        cx, cy = (config.image_size - 1) / 2 + rng.uniform(-config.center_jitter, config.center_jitter, 2)
        radius = rng.uniform(config.radius_min, config.radius_max)
        if label == 0:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        else:
            mask = (np.abs(xx - cx) <= radius) & (np.abs(yy - cy) <= radius)
        masks.append(mask)
    noise = rng.normal(0.0, config.noise_std, (n, config.image_size, config.image_size)).astype(np.float32)
    return labels, np.asarray(masks), noise


def balanced_backgrounds(labels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    background = np.empty_like(labels)
    for label in [0, 1]:
        indices = np.flatnonzero(labels == label)
        if len(indices) % 2:
            raise ValueError("Each class must have an even number of examples")
        values = np.tile(np.array([0, 1]), len(indices) // 2)
        rng.shuffle(values)
        background[indices] = values
    return background


def render(masks: np.ndarray, noise: np.ndarray, background: np.ndarray, config: Config) -> np.ndarray:
    if masks.shape != noise.shape or background.shape != (len(masks),):
        raise ValueError("Mismatched mask, noise or background shape")
    if not np.isin(background, [0, 1]).all():
        raise ValueError("Background assignments must be 0 or 1")
    base = np.where(background[:, None, None] == 0, config.dark_background, config.light_background)
    return np.clip(np.where(masks, config.foreground, base) + noise, 0, 1).astype(np.float32)


def shift_right(masks: np.ndarray, pixels: int) -> np.ndarray:
    """Shift without wrapping or cropping foreground pixels."""
    if type(pixels) is not int or not 0 <= pixels < masks.shape[2]:
        raise ValueError("Invalid horizontal shift")
    if pixels == 0:
        return masks.copy()
    if masks[:, :, -pixels:].any():
        raise ValueError("Shift would crop or wrap foreground; choose a smaller shift")
    result = np.zeros_like(masks)
    result[:, :, pixels:] = masks[:, :, :-pixels]
    return result


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def make_dataset(seed: int, config: Config) -> dict[str, np.ndarray]:
    y, masks, noise = generate_objects(seed, config.n_train, config)
    ty, tm, tn = generate_objects(seed + 10000, config.n_test, config)
    train_bg = balanced_backgrounds(y, seed + 1000)
    independent_bg = balanced_backgrounds(ty, seed + 20000)
    shifted = shift_right(tm, config.shift_pixels)
    data = {
        "train_labels": y, "train_masks": masks, "train_noise": noise,
        "train_biased_backgrounds": y.copy(), "train_varied_backgrounds": train_bg,
        "train_biased": render(masks, noise, y, config),
        "train_varied": render(masks, noise, train_bg, config),
        "test_labels": ty, "test_masks": tm, "test_noise": tn,
        "test_independent_backgrounds": independent_bg, "shifted_masks": shifted,
        "test_same_rule": render(tm, tn, ty, config),
        "test_background_reversed": render(tm, tn, 1 - ty, config),
        "test_background_independent": render(tm, tn, independent_bg, config),
        "test_position_shifted": render(shifted, tn, ty, config),
    }
    validate_dataset(data, config)
    return data


def validate_dataset(data: dict[str, np.ndarray], config: Config) -> None:
    for prefix, count in [("train", config.n_train), ("test", config.n_test)]:
        labels = data[f"{prefix}_labels"]
        if labels.shape != (count,) or not np.array_equal(np.bincount(labels, minlength=2), [count//2]*2):
            raise ValueError("Class labels must be balanced")
    for key, image in data.items():
        if key in {"train_biased", "train_varied"} or key.startswith("test_") and key[5:] in CONDITIONS:
            count = config.n_train if key.startswith("train") else config.n_test
            if image.shape != (count, config.image_size, config.image_size):
                raise ValueError(f"Wrong image shape: {key}")
            if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError(f"Invalid pixel values: {key}")
    mask = data["test_masks"]
    a, b = data["test_same_rule"], data["test_background_reversed"]
    if not np.array_equal(a[mask], b[mask]):
        raise ValueError("Background intervention changed foreground")
    for labels_key, background_key in [("train_labels", "train_varied_backgrounds"),
                                        ("test_labels", "test_independent_backgrounds")]:
        labels, bg = data[labels_key], data[background_key]
        for label in [0, 1]:
            if not np.array_equal(np.bincount(bg[labels == label], minlength=2), [len(labels)//4]*2):
                raise ValueError("Backgrounds must be balanced within each class")
    train_hashes = {array_hash(im) for key in ["train_biased", "train_varied"] for im in data[key]}
    test_hashes = {array_hash(im) for condition in CONDITIONS for im in data[f"test_{condition}"]}
    if not train_hashes.isdisjoint(test_hashes):
        raise ValueError("Exact images overlap between training and testing")
    if not np.array_equal(data["test_masks"].sum(axis=(1, 2)), data["shifted_masks"].sum(axis=(1, 2))):
        raise ValueError("Position shift changed object area")


def save_dataset(path: Path, data: dict[str, np.ndarray], seed: int, config: Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps({"seed": seed, "config": config.to_dict(), "label_names": ["circle", "square"]})
    np.savez_compressed(path, **data, metadata_json=np.array(metadata))


def load_dataset(path: Path, seed: int, config: Config) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"]))
        if metadata["seed"] != seed or metadata["config"] != config.to_dict():
            raise ValueError("Dataset configuration differs; generate into a fresh run directory")
        data = {key: saved[key] for key in saved.files if key != "metadata_json"}
    validate_dataset(data, config)
    return data
