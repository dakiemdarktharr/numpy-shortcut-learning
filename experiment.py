from pathlib import Path
import numpy as np


folder = Path(__file__).resolve().parent


def make_shapes(seed):
    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], 600)
    rng.shuffle(labels)
    yy, xx = np.mgrid[:24, :24]
    masks = []
    for label in labels:
        cx, cy = 11.5 + rng.uniform(-0.65, 0.65, 2)
        radius = rng.uniform(4.55, 5.15)
        if label == 0:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        else:
            mask = (abs(xx - cx) <= radius) & (abs(yy - cy) <= radius)
        masks.append(mask)
    noise = rng.normal(0, 0.012, (1200, 24, 24)).astype(np.float32)
    return labels, np.array(masks), noise


def balanced_backgrounds(labels, seed):
    rng = np.random.default_rng(seed)
    backgrounds = np.empty_like(labels)
    for label in [0, 1]:
        indices = np.flatnonzero(labels == label)
        values = np.tile([0, 1], len(indices) // 2)
        rng.shuffle(values)
        backgrounds[indices] = values
    return backgrounds


def make_images(masks, noise, backgrounds):
    backgrounds = np.where(backgrounds[:, None, None] == 0, 0.18, 0.82)
    return np.clip(np.where(masks, 0.02, backgrounds) + noise, 0, 1).astype(np.float32)


def make_dataset(path):
    train_y, train_masks, train_noise = make_shapes(101)
    test_y, test_masks, test_noise = make_shapes(10101)
    shifted = np.zeros_like(test_masks)
    shifted[:, :, 6:] = test_masks[:, :, :-6]
    assert np.array_equal(shifted.sum((1, 2)), test_masks.sum((1, 2)))
    np.savez_compressed(
        path,
        train_y=train_y,
        test_y=test_y,
        train_a=make_images(train_masks, train_noise, train_y),
        train_b=make_images(train_masks, train_noise, balanced_backgrounds(train_y, 1101)),
        test_original=make_images(test_masks, test_noise, test_y),
        test_reversed=make_images(test_masks, test_noise, 1 - test_y),
        test_mixed=make_images(test_masks, test_noise, balanced_backgrounds(test_y, 20101)),
        test_shifted=make_images(shifted, test_noise, test_y),
    )


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -40, 40)))


def train(images, labels):
    x = images.reshape(len(images), -1) - 0.5
    y = labels.astype(np.float32)
    weights = np.zeros(x.shape[1], dtype=np.float32)
    bias = 0.0
    for step in range(800):
        error = sigmoid(x @ weights + bias) - y
        weights -= 0.025 * (x.T @ error / len(x) + 0.001 * weights)
        bias -= 0.025 * float(error.mean())
    return weights, bias


def predict(images, weights, bias):
    x = images.reshape(len(images), -1) - 0.5
    return (sigmoid(x @ weights + bias) >= 0.5).astype(int)


if __name__ == "__main__":
    path = folder / "dataset.npz"
    if not path.exists():
        print("Creating dataset...")
        make_dataset(path)
    with np.load(path, allow_pickle=False) as file:
        data = dict(file)
    print("Training model A: biased backgrounds")
    wa, ba = train(data["train_a"], data["train_y"])
    print("Training model B: balanced backgrounds")
    wb, bb = train(data["train_b"], data["train_y"])
    np.savez(folder / "models.npz", wa=wa, ba=ba, wb=wb, bb=bb)
    rows = ["test,model_a_correct,model_b_correct,total"]
    print(f"\n{'Test':<14} {'Model A':>12} {'Model B':>12}")
    for name in ["original", "reversed", "mixed", "shifted"]:
        images = data["test_" + name]
        a = int(np.sum(predict(images, wa, ba) == data["test_y"]))
        b = int(np.sum(predict(images, wb, bb) == data["test_y"]))
        n = len(images)
        print(f"{name:<14} {100 * a / n:11.2f}% {100 * b / n:11.2f}%")
        rows.append(f"{name},{a},{b},{n}")
    (folder / "results.csv").write_text("\n".join(rows), encoding="utf-8")
    print("\nSaved: dataset.npz, models.npz, results.csv")
