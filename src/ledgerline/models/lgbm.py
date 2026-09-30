"""LightGBM fraud model."""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl

from ledgerline.features.pipeline import CategoryEncoder

SEED = 42
PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.1,
    "num_leaves": 255,
    "min_child_samples": 50,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": SEED,
    # Same data + same thread count => bit-identical model.
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 8,
    "verbose": -1,
}
MAX_ROUNDS = 5_000
EARLY_STOPPING_ROUNDS = 100


@dataclass
class FraudModel:
    booster: lgb.Booster
    encoder: CategoryEncoder
    features: list[str]

    def score(self, df: pl.DataFrame) -> np.ndarray:
        return self.booster.predict(self.encoder.transform(df, self.features))


def train_lgbm(
    train: pl.DataFrame,
    val: pl.DataFrame,
    features: list[str],
    balance_classes: bool = False,
    seed: int = SEED,
) -> FraudModel:
    """Fit on train; validation only decides when to stop adding trees."""
    encoder = CategoryEncoder().fit(train, features)
    y_train = train["isFraud"].to_numpy()
    params = {**PARAMS, "seed": seed}
    if balance_classes:
        params["scale_pos_weight"] = float((y_train == 0).sum() / (y_train == 1).sum())

    categorical = encoder.categorical(features)
    train_set = lgb.Dataset(
        encoder.transform(train, features),
        y_train,
        feature_name=features,
        categorical_feature=categorical,
        free_raw_data=True,
    )
    val_set = lgb.Dataset(
        encoder.transform(val, features),
        val["isFraud"].to_numpy(),
        reference=train_set,
        categorical_feature=categorical,
    )
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return FraudModel(booster, encoder, features)
