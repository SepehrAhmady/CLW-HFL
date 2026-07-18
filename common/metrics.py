"""Shared metrics utilities.

Used both for each layer's own-subset validation metrics (Part 3, section
3.2) and for the unified global confusion matrix in evaluate_global.py
(Part 6, step 4). Keeping a single implementation means Table I, Figure 3,
Figure 4, and Figure 5 are all guaranteed to be computed the same way.
"""
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)


def compute_metrics(y_true, y_pred, labels=None):
    """Returns a dict with accuracy, macro/weighted P-R-F1, per-class F1,
    and the raw confusion matrix (as a list of lists, JSON-serializable).

    `labels` should be passed explicitly (e.g. list(range(10)) for the
    global evaluator) so that classes absent from a particular batch still
    appear as all-zero rows/columns rather than silently shifting indices.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": float(acc),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
        "per_class": {
            str(label): {
                "precision": float(p_per_class[i]),
                "recall": float(r_per_class[i]),
                "f1": float(f1_per_class[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": cm.tolist(),
        "labels": list(labels),
    }


def aggregate_seeds(metric_dicts):
    """Given a list of compute_metrics() outputs (one per seed), return
    mean +/- std for every scalar field, as required by Part 5, section 3.6
    for every table/figure.
    """
    scalar_keys = [
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted",
    ]
    out = {}
    for k in scalar_keys:
        vals = np.array([m[k] for m in metric_dicts], dtype=float)
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0)), "n_seeds": len(vals)}
    return out
