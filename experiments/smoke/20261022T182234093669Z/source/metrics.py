"""NumPy-only binary ranking metrics, grouping equal scores at one threshold.

AUROC uses trapezoids on the empirical ROC, AP uses non-interpolated recall
increments times threshold precision. No target is silently removed from macro.
"""
import numpy as np


def binary_metrics(target, score):
    y = np.asarray(target)
    s = np.asarray(score, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape or not len(y):
        raise ValueError("Expected nonempty matching binary vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError("Nonbinary labels or non-finite scores")
    order = np.argsort(-s, kind="stable")
    y = y[order].astype(np.int64)
    s = s[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s)-1]
    tp = np.cumsum(y)[ends]
    fp = ends+1-tp
    positive = int(y.sum())
    negative = len(y)-positive
    auc = ap = None
    if positive:
        recall = tp/positive
        ap = float(np.sum(np.diff(np.r_[0.0, recall]) * tp/(ends+1)))
    if positive and negative:
        tpr = np.r_[0.0, tp/positive]
        fpr = np.r_[0.0, fp/negative]
        auc = float(np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])*0.5))
    return dict(auroc=auc, average_precision=ap, positive=positive, negative=negative)


def multilabel_metrics(target, score, classes):
    y, s = np.asarray(target), np.asarray(score)
    if y.ndim != 2 or y.shape != s.shape or y.shape[1] != len(classes):
        raise ValueError("Expected matching recording-by-label arrays")
    per_label = {c: binary_metrics(y[:, i], s[:, i]) for i, c in enumerate(classes)}
    def macro(key):
        values = [m[key] for m in per_label.values()]
        return None if any(v is None for v in values) else float(np.mean(values))
    return dict(macro_auroc=macro("auroc"), macro_average_precision=macro("average_precision"), per_label=per_label)


def selection_score(metrics, weights):
    if set(metrics) != set(weights) or not np.isclose(sum(weights.values()), 1):
        raise ValueError("Configuration/weight mismatch")
    values = [metrics[k]["macro_auroc"] for k in weights]
    if any(v is None or not np.isfinite(v) for v in values):
        return None
    return float(sum(weights[k]*metrics[k]["macro_auroc"] for k in weights))


def improves(score, best):
    return score is not None and np.isfinite(score) and (best is None or score > best)
