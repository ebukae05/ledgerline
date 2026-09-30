import numpy as np
import polars as pl
import pytest

from ledgerline.features.pipeline import CategoryEncoder
from ledgerline.models.lgbm import train_lgbm
from ledgerline.models.registry import load_model, model_version, save_model


def test_encoder_codes_come_from_train_and_unseen_values_become_missing():
    train = pl.DataFrame({"product": ["W", "W", "C", None], "amount": [1.0, 2.0, 3.0, 4.0]})
    later = pl.DataFrame({"product": ["C", "Z", None], "amount": [5.0, 6.0, 7.0]})

    encoder = CategoryEncoder().fit(train, ["product", "amount"])
    matrix = encoder.transform(later, ["product", "amount"])

    assert encoder.vocab == {"product": ["W", "C"]}  # most frequent first
    assert encoder.categorical(["product", "amount"]) == [0]
    assert matrix[0].tolist() == [1.0, 5.0]
    assert np.isnan(matrix[1, 0]) and np.isnan(matrix[2, 0])


@pytest.fixture(scope="module")
def tiny_model():
    rng = np.random.default_rng(0)
    n = 4_000
    risky = rng.random(n) < 0.3
    df = pl.DataFrame(
        {
            "isFraud": (rng.random(n) < np.where(risky, 0.3, 0.02)).astype(np.int8),
            "product": np.where(risky, "C", "W"),
            "amount": rng.uniform(1, 500, n),
        }
    )
    train, val = df.slice(0, 3_000), df.slice(3_000)
    return train_lgbm(train, val, ["product", "amount"]), val


def test_lgbm_learns_the_signal(tiny_model):
    model, val = tiny_model
    scores = model.score(val)
    risky = (val["product"] == "C").to_numpy()

    assert scores[risky].mean() > 2 * scores[~risky].mean()


def test_save_and_load_give_identical_scores(tiny_model, tmp_path):
    model, val = tiny_model

    save_model(model, {"threshold": 0.5}, models_dir=tmp_path)
    loaded, meta = load_model(models_dir=tmp_path)

    assert meta["version"] == model_version(model)
    assert meta["threshold"] == 0.5
    assert np.array_equal(loaded.score(val), model.score(val))


def test_version_is_a_hash_of_the_trees(tiny_model):
    model, _ = tiny_model
    assert model_version(model) == model_version(model)
    assert model_version(model).startswith("lgbm-")


def test_missing_model_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m ledgerline.train"):
        load_model(models_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="not found"):
        load_model("lgbm-doesnotexist", models_dir=tmp_path)
