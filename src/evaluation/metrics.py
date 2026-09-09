from __future__ import annotations

import math
from typing import Any

from sklearn.metrics import average_precision_score, roc_auc_score


def intish(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def floatish(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def maybe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def auroc(labels: list[int], scores: list[float]) -> float | None:
    if not labels or len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def ap(labels: list[int], scores: list[float]) -> float | None:
    if not labels or sum(labels) == 0:
        return None
    return float(average_precision_score(labels, scores))


def metrics(rows: list[dict[str, Any]], gt_key: str, score_key: str) -> dict[str, Any]:
    use = [row for row in rows if row.get(gt_key) not in (None, "") and maybe_float(row.get(score_key)) is not None]
    labels = [intish(row[gt_key]) for row in use]
    scores = [floatish(row[score_key]) for row in use]
    preds = [int(score > 0.0) for score in scores]
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    return {
        "n": len(use), "pos": sum(labels), "neg": len(labels) - sum(labels),
        "AUROC": auroc(labels, scores), "AP": ap(labels, scores),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }
