"""Assemble model inputs from feature groups.

The same functions build features for training, evaluation and (in Phase 5)
the API, so the model always sees identically computed inputs.

History features are computed over the whole timeline before splitting, so a
validation transaction can see a card's training-period history, exactly as
it would in production. They only ever look backward (see history.py).
"""

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ledgerline.features.basic import basic_features
from ledgerline.features.history import HISTORY_FEATURES, history_features

# Never used as model inputs. TransactionDT/day would let the model learn
# calendar position, which can't carry over to future dates.
EXCLUDED = {"TransactionID", "isFraud", "TransactionDT", "day", "TransactionAmt"}
BASIC_GROUP = "basic"
HISTORY_GROUP = "history"
RAW_GROUP = "raw"


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    """Original columns plus basic and history features, prefixed to avoid clashes."""
    basic = basic_features(df).rename(lambda c: f"basic_{c}")
    return df.hstack(basic).hstack(history_features(df))


def feature_groups(df: pl.DataFrame) -> dict[str, list[str]]:
    """Column names in each group, for a frame produced by add_features."""
    derived = {c for c in df.columns if c.startswith("basic_")} | set(HISTORY_FEATURES)
    return {
        BASIC_GROUP: [c for c in df.columns if c.startswith("basic_")],
        HISTORY_GROUP: list(HISTORY_FEATURES),
        RAW_GROUP: [c for c in df.columns if c not in derived and c not in EXCLUDED],
    }


@dataclass
class CategoryEncoder:
    """Map text columns to integer codes learned from the training split.

    Values not seen in training, and missing values, become null, which
    LightGBM treats as missing.
    """

    vocab: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, df: pl.DataFrame, columns: list[str]) -> "CategoryEncoder":
        self.vocab = {
            c: df[c].drop_nulls().value_counts(sort=True)[c].to_list()
            for c in columns
            if df.schema[c] == pl.String
        }
        return self

    def transform(self, df: pl.DataFrame, columns: list[str]) -> np.ndarray:
        exprs = []
        for c in columns:
            if c in self.vocab:
                mapping = pl.DataFrame(
                    {c: self.vocab[c], "code": range(len(self.vocab[c]))},
                    schema={c: pl.String, "code": pl.Int32},
                )
                exprs.append(pl.col(c).replace_strict(mapping[c], mapping["code"], default=None))
            else:
                exprs.append(pl.col(c).cast(pl.Float32))
        return df.select(exprs).to_numpy().astype(np.float32)

    def categorical(self, columns: list[str]) -> list[int]:
        return [i for i, c in enumerate(columns) if c in self.vocab]
