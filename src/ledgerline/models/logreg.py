"""Logistic regression on the basic per-row features.

Category lists and scaling are learned from whatever the pipeline is fit on
(the training split), so validation and test values never shape the encoding.
"""

import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ledgerline.features.basic import CATEGORICAL, NUMERIC, basic_features

SEED = 42
# Email domains seen fewer times than this share one "infrequent" bucket, so
# the model can't memorize a domain that shows up a handful of times.
MIN_CATEGORY_COUNT = 500


def build_logreg() -> Pipeline:
    encode = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", min_frequency=MIN_CATEGORY_COUNT
                ),
                CATEGORICAL,
            ),
            ("numeric", StandardScaler(), NUMERIC),
        ]
    )
    return Pipeline(
        [
            ("encode", encode),
            ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]
    )


def fit_logreg(train: pl.DataFrame) -> Pipeline:
    return build_logreg().fit(basic_features(train), train["isFraud"].to_numpy())


def score_logreg(model: Pipeline, df: pl.DataFrame):
    return model.predict_proba(basic_features(df))[:, 1]
