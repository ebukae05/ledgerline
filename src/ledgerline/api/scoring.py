"""Score one transaction with the same feature code used in training.

Training computes features with `add_features` over the whole timeline. Here
the card's earlier transactions come from the store, the new transaction is
appended, and the same `add_features` runs on that small frame. The new row's
features only ever depend on earlier transactions of the same card, so the
result matches training exactly (tests/test_api.py and the parity test check
this).
"""

import math
import re

import numpy as np
import polars as pl

from ledgerline.api.store import HISTORY_SCHEMA
from ledgerline.features.basic import basic_features
from ledgerline.features.history import HISTORY_FEATURES, history_features
from ledgerline.models.lgbm import FraudModel

TOP_REASONS = 3
# What the data provider disclosed about masked column families (Kaggle data page).
MASKED_FAMILIES = {
    "C": "count, e.g. addresses linked to the card",
    "D": "time gap, e.g. days since a previous transaction",
    "M": "match check, e.g. name on card vs address",
    "V": "provider-engineered risk feature",
    "id_": "identity/device signal",
}
NAMED_IDENTITY = {
    "id_30": "operating system",
    "id_31": "browser",
    "id_33": "screen resolution",
    "DeviceInfo": "device",
    "P_emaildomain": "purchaser email domain",
    "R_emaildomain": "recipient email domain",
    "dist1": "distance between billing and other address",
    "addr1": "billing region",
    "card1": "card number bucket",
}
# Columns the feature builders read; always present in the frame.
NEEDED = {
    "TransactionID": pl.Int64,
    "TransactionDT": pl.Int64,
    "TransactionAmt": pl.Float64,
    "ProductCD": pl.String,
    "card1": pl.Int32,
    "card4": pl.String,
    "card6": pl.String,
    "addr1": pl.Float32,
    "D1": pl.Float32,
    "DeviceType": pl.String,
    "DeviceInfo": pl.String,
    "P_emaildomain": pl.String,
    "id_01": pl.Float32,
}


def _duration(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f} seconds"
    if seconds < 7_200:
        return f"{seconds / 60:.0f} minutes"
    if seconds < 172_800:
        return f"{seconds / 3_600:.0f} hours"
    return f"{seconds / 86_400:.0f} days"


def describe(feature: str, value) -> str:
    """Plain-English reason for one feature's value."""
    missing = value is None or (isinstance(value, float) and math.isnan(value))
    who = "this card" if feature.startswith("card_") else "this cardholder (card + address)"
    windows = {"_n_1h": "the last hour", "_n_24h": "the last 24 hours", "_n_7d": "the last 7 days"}
    for suffix, span in windows.items():
        if feature.endswith(suffix):
            return f"{value:.0f} earlier transactions by {who} in {span}"
    if feature.endswith("_n_prior"):
        if value == 0:
            return f"first transaction seen for {who}"
        return f"{value:.0f} earlier transactions by {who}"
    if feature.endswith("_secs_since_prev"):
        if missing:
            return f"no earlier transaction for {who}"
        return f"only {_duration(value)} since the previous transaction by {who}"
    if feature.endswith("_amt_vs_mean"):
        if missing:
            return f"no earlier amounts for {who} to compare with"
        return f"amount is {value:.1f}x the average for {who}"
    if feature.endswith("_seen_before"):
        what = "device" if "device" in feature else "email domain"
        if missing:
            return f"no {what} information"
        return f"{what} {'already used' if value else 'never used before'} by {who}"
    basic = {
        "basic_log_amount": lambda v: f"amount ${math.expm1(v):,.2f}",
        "basic_ProductCD": lambda v: f"product type {v}",
        "basic_card4": lambda v: f"card network {v}",
        "basic_card6": lambda v: f"card type {v}",
        "basic_DeviceType": lambda v: f"device type {v}",
        "basic_P_emaildomain": lambda v: f"email domain {v}",
        "basic_hour": lambda v: f"hour of day {v} (shifted clock)",
        "basic_has_identity": lambda v: "identity data present" if v else "no identity data",
        "basic_odd_cents": lambda v: (
            "amount has more than 2 decimals (currency conversion?)" if v else "round-cent amount"
        ),
    }
    if feature in basic:
        return basic[feature](value)
    shown = "missing" if missing else (f"{value:g}" if isinstance(value, float) else value)
    if feature in NAMED_IDENTITY:
        return f"{NAMED_IDENTITY[feature]}: {shown}"
    for prefix, family in MASKED_FAMILIES.items():
        if re.fullmatch(prefix + r"\d+", feature):
            return f"{feature} = {shown} ({family}; exact meaning masked by the data provider)"
    return f"{feature} = {shown}"


class Scorer:
    def __init__(self, model: FraudModel, meta: dict, threshold: float | None = None):
        self.model = model
        self.version = meta["version"]
        self.threshold = float(meta["threshold"] if threshold is None else threshold)
        derived = set(HISTORY_FEATURES) | {f for f in model.features if f.startswith("basic_")}
        self.raw_features = [f for f in model.features if f not in derived]
        strings = set(model.encoder.vocab)
        self.schema = dict(NEEDED)
        for f in self.raw_features:
            self.schema.setdefault(f, pl.String if f in strings else pl.Float32)

    def features(self, tx: dict, history: pl.DataFrame) -> pl.DataFrame:
        """Feature row for `tx`, computed exactly as in training.

        Same functions as `add_features`, applied to the minimum data: history
        features over just the history columns of past + current transaction,
        basic features over the current row (they never look at other rows).
        tests/test_api.py and tests/test_parity_real.py check the result is
        identical to `add_features` over the full timeline.
        """
        row = pl.DataFrame([{c: tx.get(c) for c in self.schema}], schema=self.schema)
        past = history.filter(pl.col("TransactionDT") < tx["TransactionDT"])
        timeline = pl.concat(
            [past.cast(HISTORY_SCHEMA), row.select(list(HISTORY_SCHEMA)).cast(HISTORY_SCHEMA)]
        )
        basic = basic_features(row).rename(lambda c: f"basic_{c}")
        return row.hstack(basic).hstack(history_features(timeline).tail(1))

    def score(self, tx: dict, history: pl.DataFrame) -> tuple[float, bool, list[dict]]:
        feats = self.features(tx, history)
        matrix = self.model.encoder.transform(feats, self.model.features)
        # One thread per prediction: a single row gains nothing from more, and
        # with several requests in flight extra threads just compete.
        score = float(self.model.booster.predict(matrix, num_threads=1)[0])
        flagged = score >= self.threshold
        # Exact per-feature contributions cost ~0.7 s on this model, so they're
        # computed only for flagged transactions, which is where a reason is owed.
        reasons = self.reasons(feats, matrix) if flagged else []
        return score, flagged, reasons

    def reasons(self, feats: pl.DataFrame, matrix: np.ndarray) -> list[dict]:
        contributions = self.model.booster.predict(matrix, pred_contrib=True, num_threads=1)[0][:-1]
        top = np.argsort(contributions)[::-1][:TOP_REASONS]
        out = []
        for i in top:
            if contributions[i] <= 0:
                break
            name = self.model.features[i]
            value = feats[name][0]
            out.append(
                {
                    "feature": name,
                    "value": value,
                    "contribution": float(contributions[i]),
                    "text": describe(name, value),
                }
            )
        return out
