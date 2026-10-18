"""Classification metrics using NumPy, with no sklearn dependency."""
import numpy as np

__all__ = ["classification_metrics"]


def classification_metrics(labels: np.ndarray, scores: np.ndarray, logits: np.ndarray | None = None) -> dict:
    y = np.asarray(labels)
    p = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or not len(y) or p.shape != y.shape:
        raise ValueError("Expected equally sized, nonempty label/score vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError("Expected binary labels and scores in [0, 1]")
    predicted = (p >= 0.5).astype(np.int64)
    confusion = np.zeros((2, 2), dtype=np.int64)
    np.add.at(confusion, (y.astype(np.int64), predicted), 1)
    per_class = []
    for label, name in enumerate(["circle", "square"]):
        tp = int(confusion[label, label])
        support = int(confusion[label].sum())
        predicted_count = int(confusion[:, label].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        per_class.append({"label": label, "name": name, "support": support,
                          "precision": precision, "recall": recall, "f1": f1})
    if logits is not None:
        z = np.asarray(logits, dtype=np.float64)
        if z.shape != y.shape or not np.isfinite(z).all():
            raise ValueError("Invalid logits")
        log_loss = float(np.mean(np.logaddexp(0.0, z)-y*z))
    else:
        safe = np.clip(p, 1e-15, 1-1e-15)
        log_loss = float(-np.mean(y*np.log(safe)+(1-y)*np.log1p(-safe)))
    return {"n": len(y), "correct": int(np.trace(confusion)), "accuracy": float(np.trace(confusion)/len(y)),
            "precision_macro": float(np.mean([r["precision"] for r in per_class])),
            "recall_macro": float(np.mean([r["recall"] for r in per_class])),
            "f1_macro": float(np.mean([r["f1"] for r in per_class])),
            "log_loss": log_loss, "brier_score": float(np.mean((p-y)**2)),
            "confusion_matrix": confusion.tolist(), "confusion_rows": "actual circle, actual square",
            "confusion_columns": "predicted circle, predicted square", "per_class": per_class,
            "zero_division": 0, "score_status": "uncalibrated sigmoid output"}
