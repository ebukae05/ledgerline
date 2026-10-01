"""Classification metrics for merchant categories."""

from collections import Counter

from sklearn.metrics import accuracy_score, f1_score

from ledgerline.merchants.data import CATEGORIES


def classification_report(y_true: list[str], y_pred: list[str], top_confusions: int = 5) -> dict:
    """Accuracy, macro-F1 (each category counts equally) and the commonest mistakes."""
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must be the same non-zero length")
    mistakes = Counter((t, p) for t, p in zip(y_true, y_pred, strict=True) if t != p)
    per_class = f1_score(y_true, y_pred, labels=CATEGORIES, average=None, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=sorted(set(y_true)), average="macro", zero_division=0)
        ),
        "f1_by_category": dict(zip(CATEGORIES, map(float, per_class), strict=True)),
        "top_confusions": [
            {"true": t, "predicted": p, "count": n}
            for (t, p), n in mistakes.most_common(top_confusions)
        ],
    }
