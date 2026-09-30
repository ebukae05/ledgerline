"""Time-based train/validation/test split.

Fraud patterns drift over time, and the model will always be scoring
transactions that happen after the ones it was trained on. A random split
lets the model see the future during training, so its scores look better than
they will ever be in production. Splitting by time mirrors real deployment.
"""

from typing import NamedTuple

import polars as pl


class Splits(NamedTuple):
    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame


def time_split(
    df: pl.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    time_col: str = "TransactionDT",
) -> Splits:
    """Split rows into train, validation and test by time, oldest first.

    Cutoffs are timestamps, not row positions, so rows that share a timestamp
    always land in the same split. That guarantees every validation row is
    strictly later than every training row, and every test row strictly later
    than every validation row. Split sizes are therefore close to, not exactly,
    the requested fractions.
    """
    if time_col not in df.columns:
        raise ValueError(f"column {time_col!r} not in DataFrame")
    if df[time_col].null_count():
        raise ValueError(f"column {time_col!r} has nulls; every row needs a time")
    if not (0 < train_frac and 0 < val_frac and train_frac + val_frac < 1):
        raise ValueError(
            "need train_frac > 0, val_frac > 0 and train_frac + val_frac < 1, "
            f"got {train_frac} and {val_frac}"
        )

    times = df[time_col].sort()
    n = len(times)
    val_start = times[int(n * train_frac)]
    test_start = times[int(n * (train_frac + val_frac))]

    t = pl.col(time_col)
    splits = Splits(
        train=df.filter(t < val_start).sort(time_col),
        val=df.filter((t >= val_start) & (t < test_start)).sort(time_col),
        test=df.filter(t >= test_start).sort(time_col),
    )
    for name, part in splits._asdict().items():
        if part.is_empty():
            raise ValueError(f"{name} split is empty; too few distinct times in {time_col!r}")
    return splits
