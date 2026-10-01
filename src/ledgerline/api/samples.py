"""Pick example transactions for the web UI.

    python -m ledgerline.api.samples

Scores the test split offline and saves a few of each kind to
artifacts/demo/samples.json: normal transactions, frauds the model flags, and
frauds it misses (shown so the UI doesn't oversell the model). The file holds
rows of the Kaggle data, so it lives in gitignored artifacts/ like the models.
"""

import json
import math

import polars as pl

from ledgerline.api.scoring import Scorer
from ledgerline.data.load import PROJECT_ROOT, load_train
from ledgerline.data.split import time_split
from ledgerline.features.pipeline import add_features
from ledgerline.models.registry import load_model

SAMPLES_PATH = PROJECT_ROOT / "artifacts" / "demo" / "samples.json"
PER_KIND = 12
SEED = 7


def _payload(row: dict, columns: set[str]) -> dict:
    return {
        k: v
        for k, v in row.items()
        if k in columns and v is not None and not (isinstance(v, float) and math.isnan(v))
    }


def main() -> None:
    model, meta = load_model()
    columns = set(Scorer(model, meta).schema) | {"TransactionID", "TransactionDT"}
    _, _, test = time_split(add_features(load_train()))
    test = test.with_columns(pl.Series("score", model.score(test)))
    flagged = pl.col("score") >= meta["threshold"]
    fraud = pl.col("isFraud") == 1
    kinds = {
        "normal": test.filter(~fraud & ~flagged),
        "fraud_flagged": test.filter(fraud & flagged),
        "fraud_missed": test.filter(fraud & ~flagged),
    }
    out = {
        "model_version": meta["version"],
        **{
            kind: [
                _payload(row, columns)
                for row in rows.sample(min(PER_KIND, rows.height), seed=SEED).iter_rows(named=True)
            ]
            for kind, rows in kinds.items()
        },
    }
    SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLES_PATH.write_text(json.dumps(out) + "\n")
    print(
        f"Saved {sum(len(v) for k, v in out.items() if k != 'model_version')} samples to "
        f"{SAMPLES_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
