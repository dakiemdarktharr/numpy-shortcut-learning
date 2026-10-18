"""Logistic regression, binary cross entropy and full-batch gradients from scratch."""
import json
from pathlib import Path
import numpy as np

__all__ = ["sigmoid", "loss_and_gradients", "LogisticRegression"]


def sigmoid(logits: np.ndarray) -> np.ndarray:
    # Clipping avoids overflow; no ML library implements this operation for us.
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))


def loss_and_gradients(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, l2: float) -> tuple:
    logits = x @ w + b
    error = sigmoid(logits) - y
    loss = np.mean(np.logaddexp(0.0, logits) - y * logits) + 0.5 * l2 * np.sum(w*w)
    return float(loss), x.T @ error / len(x) + l2*w, float(error.mean())


class LogisticRegression:
    """576 raw-pixel weights and one bias; square is the positive class."""

    def __init__(self, steps: int = 800, learning_rate: float = 0.025, l2: float = 0.001, history_every: int = 25):
        if steps < 1 or learning_rate <= 0 or l2 < 0 or history_every < 1:
            raise ValueError("Invalid optimizer settings")
        self.steps, self.learning_rate, self.l2, self.history_every = steps, learning_rate, l2, history_every
        self.weights: np.ndarray | None = None
        self.bias = 0.0
        self.history: list[dict] = []

    @staticmethod
    def _features(images: np.ndarray) -> np.ndarray:
        values = np.asarray(images, dtype=np.float32)
        if values.ndim < 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Images must be a nonempty finite batch")
        if values.min() < 0 or values.max() > 1:
            raise ValueError("Pixels must lie in [0, 1]")
        return values.reshape(len(values), -1) - 0.5

    def fit(self, images: np.ndarray, labels: np.ndarray) -> "LogisticRegression":
        x = self._features(images)
        y = np.asarray(labels, dtype=np.float32)
        if y.shape != (len(x),) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
            raise ValueError("Training labels must contain both classes and match images")
        self.weights = np.zeros(x.shape[1], dtype=np.float32)
        self.bias = 0.0
        self.history = []
        # Same float32 update order throughout the reference protocol.
        for step in range(self.steps + 1):
            logits = x @ self.weights + self.bias
            if step % self.history_every == 0 or step == self.steps:
                data_loss = float(np.mean(np.logaddexp(0.0, logits) - y * logits))
                penalty = 0.5 * self.l2 * float(np.sum(self.weights*self.weights))
                self.history.append({"step": step, "data_loss": data_loss, "objective": data_loss + penalty})
            if step == self.steps:
                break
            error = sigmoid(logits) - y
            self.weights -= self.learning_rate * (x.T @ error / len(x) + self.l2 * self.weights)
            self.bias -= self.learning_rate * float(error.mean())
        if not np.isfinite(self.weights).all() or not np.isfinite(self.bias):
            raise FloatingPointError("Non-finite fitted parameters")
        return self

    def decision_function(self, images: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model is not fitted")
        x = self._features(images)
        if x.shape[1] != len(self.weights):
            raise ValueError("Image feature count differs from model")
        return x @ self.weights + self.bias

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        """Uncalibrated sigmoid scores for square, not a confidence guarantee."""
        return sigmoid(self.decision_function(images))

    def predict(self, images: np.ndarray) -> np.ndarray:
        return (self.predict_proba(images) >= 0.5).astype(np.int64)

    def save(self, path: Path, metadata: dict) -> None:
        if self.weights is None:
            raise ValueError("Model is not fitted")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, weights=self.weights, bias=np.array(self.bias, dtype=np.float64),
                            metadata_json=np.array(json.dumps(metadata)))

    @classmethod
    def load(cls, path: Path) -> tuple["LogisticRegression", dict]:
        with np.load(path, allow_pickle=False) as saved:
            weights, bias = saved["weights"], float(saved["bias"])
            metadata = json.loads(str(saved["metadata_json"]))
        if weights.ndim != 1 or not np.isfinite(weights).all() or not np.isfinite(bias):
            raise ValueError("Invalid model parameters")
        result = cls()
        result.weights, result.bias = weights, bias
        return result, metadata
