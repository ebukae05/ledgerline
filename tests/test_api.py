"""API tests against a tiny model trained with the real feature pipeline."""

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from ledgerline.api.app import MerchantService, Settings, StartupError, create_app
from ledgerline.api.scoring import Scorer, describe
from ledgerline.api.store import HISTORY_SCHEMA, MemoryStore, input_hash
from ledgerline.data.split import time_split
from ledgerline.features.pipeline import add_features, feature_groups
from ledgerline.merchants import tfidf
from ledgerline.models.lgbm import train_lgbm
from ledgerline.models.registry import load_model, save_model

DAY = 86_400


def synthetic_transactions(n=3_000, seed=0) -> pl.DataFrame:
    """Fake transactions where fraud comes in bursts on risky cards, so history matters."""
    rng = np.random.default_rng(seed)
    card = rng.integers(1000, 1060, n)
    risky = card % 7 == 0
    return pl.DataFrame(
        {
            "TransactionID": np.arange(n, dtype=np.int64),
            "isFraud": (rng.random(n) < np.where(risky, 0.4, 0.02)).astype(np.int8),
            "TransactionDT": np.sort(rng.integers(DAY, 60 * DAY, n)).astype(np.int64),
            "TransactionAmt": np.round(rng.uniform(1, 400, n), 2),
            "ProductCD": rng.choice(["W", "C", "H"], n),
            "card1": card.astype(np.int32),
            "card4": rng.choice(["visa", "mastercard"], n),
            "card6": rng.choice(["debit", "credit"], n),
            "addr1": rng.choice([100.0, 200.0], n).astype(np.float32),
            "D1": rng.integers(0, 3, n).astype(np.float32),
            "DeviceType": rng.choice(["mobile", "desktop"], n),
            "DeviceInfo": rng.choice(["iOS", "Windows", "Android"], n),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "outlook.com"], n),
            "id_01": np.where(rng.random(n) < 0.3, -5.0, np.nan).astype(np.float32),
            "C1": rng.integers(0, 5, n).astype(np.float32),
        }
    ).with_columns(pl.col("id_01").fill_nan(None), (pl.col("TransactionDT") // DAY).alias("day"))


@pytest.fixture(scope="module")
def data():
    return synthetic_transactions()


@pytest.fixture(scope="module")
def registry(tmp_path_factory, data):
    df = add_features(data)
    groups = feature_groups(df)
    features = groups["basic"] + groups["history"] + groups["raw"]
    train, val, _ = time_split(df)
    model = train_lgbm(train, val, features)
    folder = tmp_path_factory.mktemp("models")
    save_model(model, {"threshold": 0.5}, models_dir=folder)
    return folder


@pytest.fixture(scope="module")
def merchant_model():
    texts = [f"[debit] SAFEWAY #{i}" for i in range(30)] + [f"[debit] SHELL {i}" for i in range(30)]
    return tfidf.fit(texts, ["Groceries"] * 30 + ["Transportation"] * 30)


def payload(row: dict) -> dict:
    """A JSON body for one transaction: raw fields only, no label or derived columns."""
    return {k: v for k, v in row.items() if k not in ("isFraud", "day") and v is not None}


def make_client(registry, merchant_model, store=None, threshold=None, llm=None):
    model, meta = load_model(models_dir=registry)
    scorer = Scorer(model, meta, threshold)
    app = create_app(
        Settings(database_url=None),
        scorer=scorer,
        merchant=MerchantService(merchant_model, llm, "tfidf-test"),
        store=store if store is not None else MemoryStore(),
    )
    return TestClient(app), app


@pytest.fixture
def client(registry, merchant_model):
    return make_client(registry, merchant_model)


# --- health ---------------------------------------------------------------


def test_health_reports_model_and_database(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] and r.json()["database"]
    assert r.json()["model_version"].startswith("lgbm-")


def test_health_is_503_when_the_database_is_down(client):
    c, app = client
    app.state.store.fail_writes = True
    r = c.get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "degraded"


# --- scoring ----------------------------------------------------------------


def test_score_returns_decision_and_writes_audit_record(client, data):
    c, app = client
    body = payload(data.row(0, named=True))

    r = c.post("/score", json=body)

    assert r.status_code == 200
    out = r.json()
    assert 0 <= out["score"] <= 1
    assert out["threshold"] == 0.5
    assert out["flagged"] == (out["score"] >= 0.5)
    store = app.state.store
    assert len(store.decisions) == 1
    audit = store.decisions[0]
    assert audit["input"] == body
    assert audit["input_hash"] == input_hash(body)
    assert audit["model_version"] == out["model_version"]
    assert audit["score"] == out["score"]
    assert body["TransactionID"] in store.rows  # added to history for later transactions


def test_api_scores_match_the_offline_pipeline(registry, merchant_model, data):
    # Training/serving parity: score late transactions through the API, with
    # the store holding every earlier transaction, and compare with scoring
    # the same rows through the offline pipeline used for evaluation.
    model, _ = load_model(models_dir=registry)
    offline = model.score(add_features(data))

    store = MemoryStore()
    history = data.select(list(HISTORY_SCHEMA))
    for row in history.iter_rows(named=True):
        store.add_transaction(row)
    c, _ = make_client(registry, merchant_model, store=store)

    for i in range(data.height - 40, data.height):
        r = c.post("/score", json=payload(data.row(i, named=True)))
        assert r.status_code == 200
        assert r.json()["score"] == pytest.approx(offline[i], abs=1e-12), f"row {i}"


def test_flagged_transactions_come_with_reasons(registry, merchant_model, data):
    c, _ = make_client(registry, merchant_model, threshold=1e-9)  # flag everything

    out = c.post("/score", json=payload(data.row(5, named=True))).json()

    assert out["flagged"]
    assert 1 <= len(out["reasons"]) <= 3
    assert all(reason["text"] and reason["contribution"] > 0 for reason in out["reasons"])
    contributions = [reason["contribution"] for reason in out["reasons"]]
    assert contributions == sorted(contributions, reverse=True)


def test_unflagged_transactions_skip_the_expensive_reasons(registry, merchant_model, data):
    c, _ = make_client(registry, merchant_model, threshold=0.999999)

    out = c.post("/score", json=payload(data.row(5, named=True))).json()

    assert not out["flagged"] and out["reasons"] == []


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"TransactionAmt": None}, "TransactionAmt"),
        ({"TransactionAmt": "abc"}, "TransactionAmt"),
        ({"TransactionAmt": -5.0}, "TransactionAmt"),
        ({"TransactionAmt": 0}, "TransactionAmt"),
        ({"ProductCD": "Z"}, "ProductCD"),
        ({"card6": "gold"}, "card6"),
        ({"TransactionDT": -1}, "TransactionDT"),
        ({"Amount": 10.0}, "Amount"),  # unknown field: likely a typo
        ({"C1": "lots"}, "C1"),
    ],
)
def test_bad_input_is_a_clear_422_not_a_500(client, data, change, field):
    c, app = client
    body = payload(data.row(0, named=True))
    body.update(change)
    if change.get(field, 1) is None:
        del body[field]

    r = c.post("/score", json=body)

    assert r.status_code == 422
    assert any(field in message for message in r.json()["errors"])
    assert app.state.store.decisions == []  # nothing decided, nothing audited


@pytest.mark.parametrize("body", [[1, 2], "text", None, {}])
def test_malformed_bodies_are_422(client, body):
    c, _ = client
    assert c.post("/score", json=body).status_code == 422


def test_database_outage_returns_503_and_no_unaudited_decision(client, data):
    c, app = client
    app.state.store.fail_writes = True

    r = c.post("/score", json=payload(data.row(0, named=True)))

    assert r.status_code == 503
    assert app.state.store.rows == {}


# --- batch ------------------------------------------------------------------


def test_batch_skips_bad_records_and_scores_the_rest(client, data):
    c, app = client
    good = [payload(data.row(i, named=True)) for i in (10, 11, 12)]
    bad = {**good[1], "TransactionAmt": -1}

    r = c.post("/score/batch", json={"transactions": [good[0], bad, good[2]]})

    assert r.status_code == 200
    out = r.json()
    assert (out["scored"], out["failed"]) == (2, 1)
    assert out["errors"][0]["index"] == 1
    assert out["errors"][0]["transaction_id"] == bad["TransactionID"]
    assert "TransactionAmt" in out["errors"][0]["errors"][0]
    assert len(app.state.store.decisions) == 2


def test_batch_size_is_limited(client, data):
    c, _ = client
    body = payload(data.row(0, named=True))
    assert c.post("/score/batch", json={"transactions": [body] * 1001}).status_code == 422
    assert c.post("/score/batch", json={"transactions": []}).status_code == 422


# --- merchant ---------------------------------------------------------------


class FakeLLM:
    model = "gemini-test"

    def __init__(self, answer=None, error=None):
        self.answer, self.error = answer, error

    def predict(self, texts):
        if self.error:
            raise self.error
        return [self.answer]


def test_merchant_uses_the_llm_when_available(registry, merchant_model):
    c, app = make_client(registry, merchant_model, llm=FakeLLM(answer="Education"))

    r = c.post("/merchant", json={"description": "PAYPAL *DATACAMP JYF7455"})

    assert r.json() == {"category": "Education", "confidence": None, "method": "llm"}
    assert app.state.store.decisions[0]["model_version"] == "gemini-test"


def test_merchant_falls_back_to_tfidf_when_the_llm_fails(registry, merchant_model):
    c, app = make_client(registry, merchant_model, llm=FakeLLM(error=TimeoutError()))

    r = c.post("/merchant", json={"description": "SAFEWAY #9999"})

    assert r.status_code == 200
    assert r.json()["method"] == "tfidf"
    assert r.json()["category"] == "Groceries"
    assert 0 < r.json()["confidence"] <= 1
    assert app.state.store.decisions[0]["model_version"] == "tfidf-test"


@pytest.mark.parametrize(
    "body",
    [{"description": ""}, {"description": "x" * 301}, {}, {"description": "OK", "direction": "up"}],
)
def test_merchant_rejects_bad_input(client, body):
    c, _ = client
    assert c.post("/merchant", json=body).status_code == 422


# --- startup ----------------------------------------------------------------


def test_refuses_to_start_without_a_model(tmp_path, merchant_model):
    with pytest.raises(StartupError, match="python -m ledgerline.train"):
        create_app(
            Settings(models_dir=tmp_path),
            merchant=MerchantService(merchant_model),
            store=MemoryStore(),
        )


def test_refuses_to_start_without_a_database(registry, merchant_model):
    model, meta = load_model(models_dir=registry)
    with pytest.raises(StartupError, match="DATABASE_URL"):
        create_app(
            Settings(database_url=None),
            scorer=Scorer(model, meta),
            merchant=MerchantService(merchant_model),
        )


@pytest.mark.parametrize(("value", "match"), [("abc", "must be a number"), ("1.5", "between")])
def test_refuses_an_invalid_threshold(registry, merchant_model, value, match):
    with pytest.raises(StartupError, match=match):
        create_app(
            Settings(models_dir=registry, flag_threshold=value),
            merchant=MerchantService(merchant_model),
            store=MemoryStore(),
        )


def test_threshold_can_be_changed_without_retraining(registry, merchant_model):
    app = create_app(
        Settings(models_dir=registry, flag_threshold="0.2"),
        merchant=MerchantService(merchant_model),
        store=MemoryStore(),
    )
    assert TestClient(app).get("/health").status_code == 200
    assert app.state.scorer.threshold == 0.2


# --- reason text ------------------------------------------------------------


@pytest.mark.parametrize(
    ("feature", "value", "expected"),
    [
        ("card_n_1h", 6.0, "6 earlier transactions by this card in the last hour"),
        ("uid_n_prior", 0, "first transaction seen for this cardholder (card + address)"),
        ("card_secs_since_prev", 90.0, "only 90 seconds since the previous transaction"),
        ("card_amt_vs_mean", 4.0, "amount is 4.0x the average for this card"),
        ("uid_device_seen_before", 0, "device never used before"),
        ("basic_log_amount", float(np.log1p(250.0)), "amount $250.00"),
        ("V258", 3.0, "V258 = 3 (provider-engineered risk feature"),
    ],
)
def test_reason_text(feature, value, expected):
    assert expected in describe(feature, value)


@pytest.mark.parametrize(
    ("feature", "value", "expected"),
    [
        ("C13", 0.0, "C13 = 0 (count, e.g. addresses linked to the card;"),
        ("D15", 120.0, "time gap"),
        ("M4", "M2", "match check"),
        ("id_31", "chrome generic", "browser: chrome generic"),
        ("V258", None, "V258 = missing (provider-engineered"),
    ],
)
def test_masked_features_get_their_family_description(feature, value, expected):
    assert expected in describe(feature, value)


def test_health_shows_recent_merchant_answers_by_method(registry, merchant_model):
    # A silent fallback keeps requests working; /health makes it visible.
    c, _ = make_client(registry, merchant_model, llm=FakeLLM(error=TimeoutError()))

    c.post("/merchant", json={"description": "SAFEWAY #1"})
    c.post("/merchant", json={"description": "SAFEWAY #2"})

    assert c.get("/health").json()["merchant_last_hour"] == {"tfidf": 2}


def test_recent_decisions_are_newest_first_without_full_input(client, data):
    c, _ = client
    for i in (0, 1, 2):
        c.post("/score", json=payload(data.row(i, named=True)))

    rows = c.get("/decisions?limit=2").json()

    assert [r["transaction_id"] for r in rows] == [2, 1]
    assert "input" not in rows[0] and rows[0]["input_hash"]
    assert c.get("/decisions?limit=0").status_code == 422
    assert c.get("/decisions?limit=500").status_code == 422


def test_samples_endpoint(registry, merchant_model, tmp_path):
    model, meta = load_model(models_dir=registry)
    path = tmp_path / "samples.json"
    app = create_app(
        Settings(samples_path=path),
        scorer=Scorer(model, meta),
        merchant=MerchantService(merchant_model),
        store=MemoryStore(),
    )
    c = TestClient(app)

    assert c.get("/samples").status_code == 404
    path.write_text('{"normal": [], "fraud_flagged": [], "fraud_missed": []}')
    assert c.get("/samples").json()["normal"] == []


def test_ui_is_served(client):
    c, _ = client
    assert c.get("/", follow_redirects=False).headers["location"] == "/ui/"
    r = c.get("/ui/")
    assert r.status_code == 200 and "Ledgerline" in r.text
