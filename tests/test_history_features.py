import numpy as np
import polars as pl
import pytest

from ledgerline.features.history import HISTORY_FEATURES, history_features

DAY = 86_400


def tx(rows):
    defaults = {
        "TransactionDT": DAY,
        "TransactionAmt": 10.0,
        "card1": 1000,
        "addr1": 300.0,
        "D1": 0.0,
        "DeviceInfo": None,
        "P_emaildomain": None,
    }
    schema = {
        "TransactionDT": pl.Int64,
        "TransactionAmt": pl.Float64,
        "card1": pl.Int32,
        "addr1": pl.Float32,
        "D1": pl.Float32,
        "DeviceInfo": pl.String,
        "P_emaildomain": pl.String,
    }
    return pl.DataFrame([{**defaults, **r} for r in rows], schema=schema)


def test_hand_computed_card_history():
    t0 = 10 * DAY
    df = tx(
        [
            {"TransactionDT": t0, "TransactionAmt": 10.0},
            {"TransactionDT": t0 + 600, "TransactionAmt": 30.0},  # 10 min later
            {"TransactionDT": t0 + 7_200, "TransactionAmt": 100.0},  # 2 h after first
            {"TransactionDT": t0 + 7_200, "card1": 2000},  # different card, same time
        ]
    )

    f = history_features(df)

    assert f["card_n_prior"].to_list() == [0, 1, 2, 0]
    assert f["card_n_1h"].to_list() == [0, 1, 0, 0]
    assert f["card_n_24h"].to_list() == [0, 1, 2, 0]
    assert f["card_secs_since_prev"].to_list() == [None, 600, 6_600, None]
    # Third transaction: $100 vs prior mean of $20.
    assert f["card_amt_vs_mean"].to_list() == [None, 3.0, 5.0, None]


def test_window_edges():
    t0 = 10 * DAY
    df = tx([{"TransactionDT": t0}, {"TransactionDT": t0 + 3_600}, {"TransactionDT": t0 + 3_601}])

    f = history_features(df)

    # Exactly 1 hour before is inside the window; 1 hour and 1 second is not.
    assert f["card_n_1h"].to_list() == [0, 1, 1]


def test_same_second_transactions_do_not_see_each_other():
    df = tx([{"TransactionAmt": 5.0}, {"TransactionAmt": 50.0}])

    f = history_features(df)

    assert f["card_n_prior"].to_list() == [0, 0]
    assert f["card_n_1h"].to_list() == [0, 0]


def test_seen_before_needs_a_strictly_earlier_use():
    t0 = 10 * DAY
    df = tx(
        [
            {"TransactionDT": t0, "DeviceInfo": "iOS", "P_emaildomain": "a.com"},
            {"TransactionDT": t0 + 60, "DeviceInfo": "iOS", "P_emaildomain": "b.com"},
            {"TransactionDT": t0 + 60, "DeviceInfo": "Windows", "P_emaildomain": None},
        ]
    )

    f = history_features(df)

    assert f["card_device_seen_before"].to_list() == [0, 1, 0]
    assert f["card_email_seen_before"].to_list() == [0, 0, None]


def test_uid_separates_cards_that_share_card1():
    # Same card1, but first used on different days (day - D1 differs).
    df = tx(
        [
            {"TransactionDT": 10 * DAY, "D1": 0.0},  # first used day 10
            {"TransactionDT": 11 * DAY, "D1": 1.0},  # first used day 10: same uid
            {"TransactionDT": 12 * DAY, "D1": 0.0},  # first used day 12: new uid
        ]
    )

    f = history_features(df)

    assert f["card_n_prior"].to_list() == [0, 1, 2]
    assert f["uid_n_prior"].to_list() == [0, 1, 0]


def test_output_is_in_input_order_with_all_columns():
    df = tx([{"TransactionDT": 30 * DAY}, {"TransactionDT": 10 * DAY}, {"TransactionDT": 20 * DAY}])

    f = history_features(df)

    assert f.columns == HISTORY_FEATURES
    assert f["card_n_prior"].to_list() == [2, 0, 1]


@pytest.fixture
def random_history():
    rng = np.random.default_rng(7)
    n = 3_000
    return tx(
        [
            {
                # Coarse times so plenty of transactions share a second.
                "TransactionDT": int(rng.integers(0, 60 * DAY) // 300 * 300),
                "TransactionAmt": float(rng.uniform(1, 500)),
                "card1": int(rng.integers(0, 40)),
                "addr1": float(rng.choice([100, 200, np.nan])),
                "D1": float(rng.integers(0, 5)),
                "DeviceInfo": rng.choice(["iOS", "Windows", "Android", None]),
                "P_emaildomain": rng.choice(["a.com", "b.com", None]),
            }
            for _ in range(n)
        ]
    ).sample(fraction=1.0, shuffle=True, seed=1)


def test_no_feature_uses_the_future(random_history):
    # The core leakage test. For many cutoffs t, delete every transaction after
    # t. If any feature of a transaction at or before t changes, it was
    # reading from the future.
    full = random_history.hstack(history_features(random_history))

    for cutoff in np.quantile(random_history["TransactionDT"].to_numpy(), [0.1, 0.3, 0.5, 0.8]):
        past = random_history.filter(pl.col("TransactionDT") <= cutoff)
        recomputed = past.hstack(history_features(past))
        expected = full.filter(pl.col("TransactionDT") <= cutoff)
        assert recomputed.equals(expected), f"features changed when data after {cutoff} removed"


def test_changing_a_later_transaction_changes_nothing_earlier(random_history):
    df = random_history.sort("TransactionDT")
    last_time = df["TransactionDT"].max()
    edited = df.with_columns(
        pl.when(pl.col("TransactionDT") == last_time)
        .then(pl.col("TransactionAmt") * 1000)
        .otherwise(pl.col("TransactionAmt"))
    )

    earlier = pl.col("TransactionDT") < last_time
    before = df.hstack(history_features(df)).filter(earlier)
    after = edited.hstack(history_features(edited)).filter(earlier)

    assert before.equals(after)
