import numpy as np
import pytest

from ledgerline.eval.metrics import (
    pr_auc,
    precision_at_top,
    recall_at_fpr,
    report,
    roc_auc,
    threshold_at_fpr,
)

# 2 fraud among 10. Ranked by score: F, L, F, L, L, L, L, L, L, L
Y = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
S = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]


def test_perfect_ranking_scores_one_everywhere():
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    assert pr_auc(y, s) == 1.0
    assert roc_auc(y, s) == 1.0
    assert recall_at_fpr(y, s, 0.0) == 1.0
    assert precision_at_top(y, s, 0.5) == 1.0


def test_reversed_ranking_has_zero_roc_auc():
    assert roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0


def test_pr_auc_by_hand():
    # Precision at each fraud found: 1/1 at rank 1, 2/3 at rank 3. Mean = 5/6.
    assert pr_auc(Y, S) == pytest.approx(5 / 6)


def test_roc_auc_by_hand():
    # 16 fraud/legit pairs. Fraud at rank 1 beats all 8 legit; fraud at rank 3 beats 7.
    assert roc_auc(Y, S) == pytest.approx(15 / 16)


def test_recall_at_fpr_by_hand():
    # 8 legit. Allowing 0 false positives catches only the rank-1 fraud;
    # allowing 1 of 8 (12.5%) reaches rank 3 and catches both.
    assert recall_at_fpr(Y, S, 0.0) == 0.5
    assert recall_at_fpr(Y, S, 0.01) == 0.5
    assert recall_at_fpr(Y, S, 0.125) == 1.0


def test_recall_at_fpr_never_exceeds_the_fpr_limit_with_ties():
    # All scores tied: flagging anything flags all 8 legit (FPR 100%), so at
    # a 50% limit the only allowed choice is to flag nothing.
    assert recall_at_fpr(Y, [0.5] * 10, 0.5) == 0.0


def test_precision_at_top_by_hand():
    assert precision_at_top(Y, S, 0.1) == 1.0  # top 1: fraud
    assert precision_at_top(Y, S, 0.2) == 0.5  # top 2: fraud, legit
    assert precision_at_top(Y, S, 0.3) == pytest.approx(2 / 3)


def test_precision_at_top_uses_average_of_tied_group():
    # All tied, 1 fraud in 4: picking 2 at random is 25% fraud on average.
    assert precision_at_top([1, 0, 0, 0], [0.5] * 4, 0.5) == 0.25


def test_precision_at_top_does_not_depend_on_row_order():
    y = np.array([1, 0, 0, 1, 0, 0])
    s = np.array([0.9, 0.5, 0.5, 0.5, 0.1, 0.1])
    order = np.array([5, 3, 1, 0, 4, 2])
    assert precision_at_top(y, s, 0.5) == precision_at_top(y[order], s[order], 0.5)


def test_report_has_every_metric():
    r = report(Y, S)
    assert r["n"] == 10
    assert r["fraud_rate"] == 0.2
    assert set(r) >= {"pr_auc", "roc_auc", "recall_at_1pct_fpr", "precision_at_top_1pct"}


@pytest.mark.parametrize(
    ("y", "s", "match"),
    [
        ([1, 0], [0.5], "same length"),
        ([1, 0], [0.5, float("nan")], "NaN"),
        ([0, 0], [0.1, 0.2], "both fraud and non-fraud"),
        ([2, 0], [0.1, 0.2], "only 0 and 1"),
    ],
)
def test_bad_input_is_rejected(y, s, match):
    with pytest.raises(ValueError, match=match):
        report(y, s)


def test_bad_fractions_are_rejected():
    with pytest.raises(ValueError, match="max_fpr"):
        recall_at_fpr(Y, S, 1.5)
    with pytest.raises(ValueError, match="top_frac"):
        precision_at_top(Y, S, 0.0)


def test_threshold_at_fpr_by_hand():
    # 0 of 8 legit allowed: cut at the rank-1 fraud (0.9). 1 of 8: down to rank 3 (0.7).
    assert threshold_at_fpr(Y, S, 0.0) == 0.9
    assert threshold_at_fpr(Y, S, 0.125) == 0.7


def test_threshold_at_fpr_matches_recall_at_fpr():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = rng.random(500) + y * 0.3
    t = threshold_at_fpr(y, s, 0.05)
    flagged = s >= t

    assert flagged[y == 0].mean() <= 0.05
    assert flagged[y == 1].mean() == recall_at_fpr(y, s, 0.05)


def test_operating_point_by_hand():
    from ledgerline.evaluate import operating_point

    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.2, 0.1])
    amounts = np.array([100.0, 50.0, 300.0, 20.0])

    op = operating_point(y, s, amounts, threshold=0.5)

    assert op["flag_rate"] == 0.5
    assert op["fpr"] == 0.5
    assert op["recall"] == 0.5
    assert op["precision"] == 0.5
    assert op["fraud_dollars_caught_share"] == 0.25  # $100 of $400
    assert op["fraud_dollars_per_1000_flags"] == 50_000.0  # $100 over 2 flags
