from __future__ import annotations

import math
from typing import Any

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover - exercised in minimal environments
    average_precision_score = None
    roc_auc_score = None


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
    if roc_auc_score is not None:
        return float(roc_auc_score(labels, scores))
    pos = [score for label, score in zip(labels, scores) if label == 1]
    neg = [score for label, score in zip(labels, scores) if label == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p_score in pos:
        for n_score in neg:
            if p_score > n_score:
                wins += 1.0
            elif p_score == n_score:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def ap(labels: list[int], scores: list[float]) -> float | None:
    if not labels or sum(labels) == 0:
        return None
    if average_precision_score is not None:
        return float(average_precision_score(labels, scores))
    positives = sum(labels)
    order = sorted(range(len(labels)), key=lambda idx: scores[idx], reverse=True)
    total = 0.0
    tp = 0
    idx = 0
    while idx < len(order):
        score = scores[order[idx]]
        end = idx
        group_pos = 0
        while end < len(order) and scores[order[end]] == score:
            group_pos += int(labels[order[end]] == 1)
            end += 1
        if group_pos:
            precision_at_group_end = (tp + group_pos) / end
            total += group_pos * precision_at_group_end
            tp += group_pos
        idx = end
    return total / positives


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
