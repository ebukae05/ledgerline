"""Metrics for ranking transactions by fraud risk.

Every function takes true labels (1 = fraud) and scores (higher = riskier).
None of them look at a flag threshold, so they compare models fairly before
anyone has picked an operating point.
"""

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _check(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError(
            f"y_true and scores must be 1-D and the same length, got {y.shape} and {s.shape}"
        )
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1")
    if y.min() == y.max():
        raise ValueError("y_true needs both fraud and non-fraud examples")
    return y.astype(int), s


def pr_auc(y_true, scores) -> float:
    """Area under the precision-recall curve (average precision).

    A random model scores the fraud rate (~0.035 here), so gains are easy to
    read. ROC-AUC, by contrast, starts at 0.5 and looks flattering when fraud is rare.
    """
    y, s = _check(y_true, scores)
    return float(average_precision_score(y, s))


def roc_auc(y_true, scores) -> float:
    """Chance that a random fraud case scores above a random legit one."""
    y, s = _check(y_true, scores)
    return float(roc_auc_score(y, s))


def recall_at_fpr(y_true, scores, max_fpr: float = 0.01) -> float:
    """Share of fraud caught when at most `max_fpr` of legit transactions get flagged.

    At 1% FPR: of every 100 good customers, at most 1 is wrongly flagged. The
    threshold only moves between distinct scores, so tied scores are flagged
    together and the FPR limit is never exceeded.
    """
    if not 0 <= max_fpr <= 1:
        raise ValueError(f"max_fpr must be between 0 and 1, got {max_fpr}")
    y, s = _check(y_true, scores)
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    return float(tpr[fpr <= max_fpr].max())


def threshold_at_fpr(y_true, scores, max_fpr: float = 0.01) -> float:
    """Lowest score cutoff that flags at most `max_fpr` of legit transactions.

    Flag a transaction when score >= the returned value. Pick this on
    validation, then apply it unchanged to test and production traffic.
    """
    if not 0 <= max_fpr <= 1:
        raise ValueError(f"max_fpr must be between 0 and 1, got {max_fpr}")
    y, s = _check(y_true, scores)
    fpr, _, thresholds = roc_curve(y, s, drop_intermediate=False)
    return float(thresholds[fpr <= max_fpr].min())


def precision_at_top(y_true, scores, top_frac: float = 0.01) -> float:
    """Share of the riskiest `top_frac` of transactions that are actually fraud.

    This is what a review team sees if they can only look at, say, the top 1%.
    When tied scores straddle the cutoff, the tied group counts at its average
    fraud rate (the expected result of breaking ties at random), so the answer
    never depends on row order.
    """
    if not 0 < top_frac <= 1:
        raise ValueError(f"top_frac must be in (0, 1], got {top_frac}")
    y, s = _check(y_true, scores)
    k = max(1, math.ceil(top_frac * len(s)))

    cutoff = np.sort(s)[::-1][k - 1]
    above = s > cutoff
    tied = s == cutoff
    n_from_tied = k - above.sum()
    expected_fraud = y[above].sum() + n_from_tied * y[tied].mean()
    return float(expected_fraud / k)


def report(y_true, scores) -> dict[str, float]:
    """All headline metrics for one model on one split."""
    y, s = _check(y_true, scores)
    return {
        "n": int(len(y)),
        "fraud_rate": float(y.mean()),
        "pr_auc": pr_auc(y, s),
        "roc_auc": roc_auc(y, s),
        "recall_at_1pct_fpr": recall_at_fpr(y, s, 0.01),
        "precision_at_top_1pct": precision_at_top(y, s, 0.01),
    }
