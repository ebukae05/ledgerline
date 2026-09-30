import numpy as np
import pytest

from ledgerline.models.logreg import fit_logreg, score_logreg
from ledgerline.models.rules import RulesBaseline


def test_rules_score_counts_rules_that_fire(transactions):
    rules = RulesBaseline(high_amount=1000.0, risky_domains=["outlook.com"])
    df = transactions(
        [
            {},  # nothing fires
            {"ProductCD": "C", "card6": "credit", "TransactionAmt": 10.0},  # 3 rules
            {"DeviceType": "mobile", "P_emaildomain": "outlook.com", "TransactionAmt": 5000.0},
        ]
    )

    assert rules.score(df).tolist() == [0.0, 3.0, 3.0]


def test_rules_treat_missing_values_as_not_firing(transactions):
    rules = RulesBaseline(risky_domains=["outlook.com"])
    df = transactions([{"card6": None, "DeviceType": None, "P_emaildomain": None}])

    assert rules.score(df).tolist() == [0.0]


def test_rules_learn_risky_domains_from_train_only(transactions):
    # risky.com: 1,000 rows at 10% fraud in train. safe.com: 2,000 rows at 0%.
    # Overall train fraud rate is 1/30, so risky.com clears the 2x bar.
    train = transactions(
        [{"P_emaildomain": "risky.com", "isFraud": int(i < 100)} for i in range(1000)]
        + [{"P_emaildomain": "safe.com"} for _ in range(2000)]
    )
    # In validation, only safe.com looks risky; that must not leak into the list.
    val = transactions(
        [{"P_emaildomain": "safe.com", "isFraud": 1}, {"P_emaildomain": "risky.com"}]
    )

    rules = RulesBaseline().fit(train, val)

    assert rules.risky_domains == ["risky.com"]


def test_rules_pick_the_amount_cutoff_that_ranks_validation_best(transactions):
    train = transactions([{"isFraud": 1}, {}])
    # Fraud sits at $600, legit at $300: only the $500 cutoff separates them.
    val = transactions([{"TransactionAmt": 600.0, "isFraud": 1}, {"TransactionAmt": 300.0}])

    rules = RulesBaseline().fit(train, val)

    assert rules.high_amount == 500.0


@pytest.fixture
def logreg_data(transactions):
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(2000):
        fraud = rng.random() < 0.2
        rows.append(
            {
                "isFraud": int(fraud),
                "ProductCD": "C" if fraud else rng.choice(["W", "C"], p=[0.9, 0.1]),
                "TransactionAmt": float(rng.uniform(1, 500)),
            }
        )
    return transactions(rows)


def test_logreg_learns_a_real_signal(logreg_data):
    model = fit_logreg(logreg_data)
    scores = score_logreg(model, logreg_data)
    is_c = (logreg_data["ProductCD"] == "C").to_numpy()

    assert scores[is_c].mean() > scores[~is_c].mean()


def test_logreg_is_reproducible(logreg_data):
    first = score_logreg(fit_logreg(logreg_data), logreg_data)
    second = score_logreg(fit_logreg(logreg_data), logreg_data)

    assert np.array_equal(first, second)


def test_logreg_scores_categories_it_never_saw(logreg_data, transactions):
    model = fit_logreg(logreg_data)
    unseen = transactions([{"ProductCD": "Z", "P_emaildomain": "new.example", "card4": None}])

    scores = score_logreg(model, unseen)

    assert 0 < scores[0] < 1
