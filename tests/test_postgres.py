"""PostgresStore against a real database.

Runs when TEST_DATABASE_URL is set (CI starts a Postgres service for this);
skipped otherwise. Uses its own schema so it never touches real data.
"""

import os
import uuid

import polars as pl
import pytest

from ledgerline.api.store import HISTORY_SCHEMA, PostgresStore

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")


@pytest.fixture
def store():
    import psycopg

    schema = f"test_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA {schema}")
    store = PostgresStore(f"{URL}?options=-csearch_path%3D{schema}")
    yield store
    store.pool.close()
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA {schema} CASCADE")


def tx(tid, dt, card=1000, amount=10.0):
    return {
        "TransactionID": tid,
        "TransactionDT": dt,
        "TransactionAmt": amount,
        "card1": card,
        "addr1": 100.0,
        "D1": 0.0,
        "DeviceInfo": "iOS",
        "P_emaildomain": None,
    }


def test_history_returns_only_earlier_transactions_for_the_card(store):
    for t in [tx(1, 100), tx(2, 200), tx(3, 300), tx(4, 150, card=2000)]:
        store.add_transaction(t)

    history = store.history(1000, before_dt=300)

    assert sorted(history["TransactionID"].to_list()) == [1, 2]
    assert history.schema == pl.Schema(HISTORY_SCHEMA)


def test_adding_the_same_transaction_twice_keeps_one_row(store):
    store.add_transaction(tx(1, 100))
    store.add_transaction(tx(1, 100, amount=999.0))

    history = store.history(1000, before_dt=1_000)

    assert history["TransactionAmt"].to_list() == [10.0]


def test_decisions_are_recorded_with_everything_needed_for_an_audit(store):
    store.record(
        {
            "endpoint": "/score",
            "transaction_id": 7,
            "input": {"TransactionID": 7, "TransactionAmt": 12.5},
            "input_hash": "abc",
            "output": {"score": 0.9, "reasons": [{"text": "x"}]},
            "score": 0.9,
            "threshold": 0.03,
            "flagged": True,
            "model_version": "lgbm-test",
        }
    )
    with store.pool.connection() as conn:
        row = conn.execute(
            "SELECT transaction_id, input, score, threshold, flagged, model_version, created_at "
            "FROM decisions"
        ).fetchone()

    assert row[0] == 7
    assert row[1]["TransactionAmt"] == 12.5
    assert row[2:6] == (0.9, 0.03, True, "lgbm-test")
    assert row[6] is not None


def test_seed_bulk_loads_and_skips_existing_rows(store):
    store.add_transaction(tx(1, 100))
    frame = pl.DataFrame([tx(i, 100 + i) for i in range(1, 6)], schema=HISTORY_SCHEMA)

    inserted = store.seed(frame)

    assert inserted == 4
    assert store.history(1000, before_dt=10_000).height == 5


def test_health(store):
    assert store.healthy()
