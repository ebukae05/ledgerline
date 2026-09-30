"""History features: what this card has done before the current transaction.

Leakage rule: a transaction's features use only transactions with a strictly
earlier TransactionDT for the same key. Transactions in the same second don't
see each other, and nothing ever looks forward in time. Past fraud labels are
never used either; in production those arrive weeks later as chargebacks.

Two keys:
- card: `card1` alone (card number bucket; shared by many real cardholders)
- uid:  `card1` + `addr1` + the day the card was first used (`day - D1`).
        D1 is "days since the card's first transaction", so subtracting it
        from the current day gives a date that stays fixed for one card. This
        is the best-known proxy for a customer ID in this dataset.

For serving, pass the card's recent history plus the new transaction; the
same function computes the same features.
"""

import polars as pl

WINDOWS = {"1h": 3_600, "24h": 86_400, "7d": 7 * 86_400}
KEYS = {
    "card": ["card1"],
    "uid": ["card1", "addr1", "first_day"],
}
SEEN_BEFORE = {"device": "DeviceInfo", "email": "P_emaildomain"}

T = "TransactionDT"


def _with_keys(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        ((pl.col(T) // 86_400) - pl.col("D1")).round(0).cast(pl.Int32).alias("first_day")
    )
    return df.with_columns(
        pl.concat_str(
            [pl.col(c).cast(pl.String).fill_null("NA") for c in cols], separator="_"
        ).alias(f"key_{name}")
        for name, cols in KEYS.items()
    )


def _key_features(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """Features for one key, one row per distinct (key, time)."""
    key = f"key_{name}"
    # Collapse same-second transactions: they share one window and must not
    # count each other.
    per_time = (
        df.group_by(key, T)
        .agg(pl.len().alias("n"), pl.col("TransactionAmt").sum().alias("amt"))
        .sort(key, T)
    )

    prior = per_time.with_columns(
        (pl.col("n").cum_sum().over(key) - pl.col("n")).alias(f"{name}_n_prior"),
        (pl.col("amt").cum_sum().over(key) - pl.col("amt")).alias("_amt_prior"),
        (pl.col(T) - pl.col(T).shift(1).over(key)).alias(f"{name}_secs_since_prev"),
    )

    for label, seconds in WINDOWS.items():
        window = per_time.rolling(
            index_column=T, period=f"{seconds}i", group_by=key, closed="left"
        ).agg(pl.col("n").sum().alias(f"{name}_n_{label}"))
        prior = prior.join(window, on=[key, T], how="left")

    return prior.select(
        key,
        T,
        pl.col(f"{name}_n_prior"),
        *(pl.col(f"{name}_n_{label}").fill_null(0) for label in WINDOWS),
        pl.col(f"{name}_secs_since_prev"),
        # No history means null, not 0/0 = NaN.
        pl.when(pl.col(f"{name}_n_prior") > 0)
        .then(pl.col("_amt_prior") / pl.col(f"{name}_n_prior"))
        .alias("_mean_prior"),
    )


def _seen_before(df: pl.DataFrame, name: str, what: str, column: str) -> pl.Expr:
    """1 if this key used this value at an earlier time, 0 if not, null if value missing."""
    key = f"key_{name}"
    first_seen = pl.col(T).min().over(key, column)
    return (
        pl.when(pl.col(column).is_null())
        .then(None)
        .otherwise((first_seen < pl.col(T)).cast(pl.Int8))
        .alias(f"{name}_{what}_seen_before")
    )


def history_features(df: pl.DataFrame) -> pl.DataFrame:
    """Return history features, one row per input row, in input order."""
    base = _with_keys(df).with_row_index("_row")
    out = base
    for name in KEYS:
        feats = _key_features(base, name)
        out = out.join(feats, on=[f"key_{name}", T], how="left").with_columns(
            (pl.col("TransactionAmt") / pl.col("_mean_prior")).alias(f"{name}_amt_vs_mean")
        )
        out = out.with_columns(
            _seen_before(out, name, what, column) for what, column in SEEN_BEFORE.items()
        )
    return out.sort("_row").select(HISTORY_FEATURES)


HISTORY_FEATURES = [
    f"{name}_{feature}"
    for name in KEYS
    for feature in [
        "n_prior",
        *(f"n_{label}" for label in WINDOWS),
        "secs_since_prev",
        "amt_vs_mean",
        *(f"{what}_seen_before" for what in SEEN_BEFORE),
    ]
]
