"""Per-row features for the baseline models.

Every feature here is computed from the transaction's own columns only, never
from other transactions, so none of them can leak information across time.
History-based features (velocity, card averages) come in Phase 3.
"""

import polars as pl

SECONDS_PER_HOUR = 3_600
HOURS_PER_DAY = 24

CATEGORICAL = ["ProductCD", "card4", "card6", "DeviceType", "P_emaildomain", "hour"]
NUMERIC = ["log_amount", "has_identity", "odd_cents"]
MISSING = "missing"


def basic_features(df: pl.DataFrame) -> pl.DataFrame:
    """Return the baseline feature table, one row per input row, same order."""
    amount = pl.col("TransactionAmt")
    return df.select(
        pl.col("ProductCD", "card4", "card6", "DeviceType", "P_emaildomain").fill_null(MISSING),
        ((pl.col("TransactionDT") // SECONDS_PER_HOUR) % HOURS_PER_DAY)
        .cast(pl.String)
        .alias("hour"),
        amount.log1p().alias("log_amount"),
        pl.col("id_01").is_not_null().cast(pl.Int8).alias("has_identity"),
        # More than 2 decimal places, e.g. 31.953. Likely a currency conversion.
        (((amount * 100).round(6) % 1).abs() > 1e-6).cast(pl.Int8).alias("odd_cents"),
    )
