"""Training/serving parity with the real model, real data and real Postgres.

    LEDGERLINE_REAL_PARITY=1 TEST_DATABASE_URL=... pytest tests/test_parity_real.py

Needs the Kaggle data, a trained model and a database seeded with
`python -m ledgerline.api.seed`, so it's skipped in CI and run locally.
Scores test-split transactions through the HTTP API (history read from
Postgres) and checks each score matches the offline pipeline that produced
the reported test metrics.
"""

import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    not (os.environ.get("LEDGERLINE_REAL_PARITY") and os.environ.get("TEST_DATABASE_URL")),
    reason="set LEDGERLINE_REAL_PARITY=1 and TEST_DATABASE_URL to run",
)
N = 300


def test_api_matches_offline_scores_on_real_test_transactions():
    from ledgerline.api.app import MerchantService, Settings, create_app
    from ledgerline.api.scoring import Scorer
    from ledgerline.api.store import PostgresStore
    from ledgerline.data.load import load_train
    from ledgerline.data.split import time_split
    from ledgerline.features.pipeline import add_features
    from ledgerline.models.registry import load_model

    model, meta = load_model()
    _, _, test = time_split(add_features(load_train()))
    sample = test.sample(N, seed=0)
    offline = model.score(sample)

    store = PostgresStore(os.environ["TEST_DATABASE_URL"])
    app = create_app(
        Settings(),
        scorer=Scorer(model, meta),
        merchant=MerchantService(model=None),
        store=store,
    )
    client = TestClient(app)
    raw_columns = set(Scorer(model, meta).schema) | {"TransactionID", "TransactionDT"}

    api = []
    for row in sample.iter_rows(named=True):
        body = {
            k: v
            for k, v in row.items()
            if k in raw_columns and v is not None and not (isinstance(v, float) and np.isnan(v))
        }
        response = client.post("/score", json=body)
        assert response.status_code == 200, response.text
        api.append(response.json()["score"])

    np.testing.assert_allclose(api, offline, rtol=0, atol=1e-9)
