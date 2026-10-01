"""Where the API keeps card history and the decision audit log.

Postgres in production. MemoryStore has the same interface for tests and
quick local runs; it is never used when DATABASE_URL is set.
"""

import hashlib
import json
import threading
from datetime import UTC, datetime
from typing import Any, Protocol

import polars as pl

# Only the columns history features read. Same names as the training data.
HISTORY_SCHEMA = {
    "TransactionID": pl.Int64,
    "TransactionDT": pl.Int64,
    "TransactionAmt": pl.Float64,
    "card1": pl.Int32,
    "addr1": pl.Float32,
    "D1": pl.Float32,
    "DeviceInfo": pl.String,
    "P_emaildomain": pl.String,
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id  BIGINT PRIMARY KEY,
    transaction_dt  BIGINT NOT NULL,
    amount          DOUBLE PRECISION NOT NULL,
    card1           INTEGER NOT NULL,
    addr1           REAL,
    d1              REAL,
    device_info     TEXT,
    p_emaildomain   TEXT
);
CREATE INDEX IF NOT EXISTS transactions_card_time ON transactions (card1, transaction_dt);

CREATE TABLE IF NOT EXISTS decisions (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    endpoint        TEXT NOT NULL,
    transaction_id  BIGINT,
    input           JSONB NOT NULL,
    input_hash      TEXT NOT NULL,
    output          JSONB NOT NULL,
    score           DOUBLE PRECISION,
    threshold       DOUBLE PRECISION,
    flagged         BOOLEAN,
    model_version   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS decisions_transaction ON decisions (transaction_id);
"""

COLUMNS = [
    "transaction_id",
    "transaction_dt",
    "amount",
    "card1",
    "addr1",
    "d1",
    "device_info",
    "p_emaildomain",
]
TO_FRAME = dict(zip(COLUMNS, HISTORY_SCHEMA, strict=True))


def input_hash(payload: dict) -> str:
    """Stable hash of a request body: same input, same hash, whatever the key order."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _history_row(tx: dict) -> tuple:
    return (
        tx["TransactionID"],
        tx["TransactionDT"],
        tx["TransactionAmt"],
        tx["card1"],
        tx.get("addr1"),
        tx.get("D1"),
        tx.get("DeviceInfo"),
        tx.get("P_emaildomain"),
    )


class Store(Protocol):
    def history(self, card1: int, before_dt: int) -> pl.DataFrame: ...
    def add_transaction(self, tx: dict) -> None: ...
    def record(self, decision: dict) -> None: ...
    def healthy(self) -> bool: ...


class MemoryStore:
    def __init__(self):
        self.rows: dict[int, tuple] = {}
        self.decisions: list[dict] = []
        self.fail_writes = False  # lets tests simulate a database outage
        self._lock = threading.Lock()

    def history(self, card1: int, before_dt: int) -> pl.DataFrame:
        with self._lock:
            rows = [r for r in self.rows.values() if r[3] == card1 and r[1] < before_dt]
        return pl.DataFrame(rows, schema=HISTORY_SCHEMA, orient="row")

    def add_transaction(self, tx: dict) -> None:
        with self._lock:
            self.rows.setdefault(tx["TransactionID"], _history_row(tx))

    def record(self, decision: dict) -> None:
        if self.fail_writes:
            raise ConnectionError("database unavailable")
        with self._lock:
            self.decisions.append({**decision, "created_at": datetime.now(UTC)})

    def healthy(self) -> bool:
        return not self.fail_writes


class PostgresStore:
    def __init__(self, url: str, pool_size: int = 10):
        from psycopg_pool import ConnectionPool

        self.pool = ConnectionPool(url, min_size=1, max_size=pool_size, open=True, timeout=5)
        with self.pool.connection() as conn:
            conn.execute(SCHEMA_SQL)

    def history(self, card1: int, before_dt: int) -> pl.DataFrame:
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"SELECT {', '.join(COLUMNS)} FROM transactions "
                "WHERE card1 = %s AND transaction_dt < %s",
                (card1, before_dt),
            ).fetchall()
        return pl.DataFrame(rows, schema=HISTORY_SCHEMA, orient="row")

    def add_transaction(self, tx: dict) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                f"INSERT INTO transactions ({', '.join(COLUMNS)}) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                _history_row(tx),
            )

    def record(self, decision: dict[str, Any]) -> None:
        from psycopg.types.json import Jsonb

        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO decisions (endpoint, transaction_id, input, input_hash, output, "
                "score, threshold, flagged, model_version) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    decision["endpoint"],
                    decision.get("transaction_id"),
                    Jsonb(decision["input"]),
                    decision["input_hash"],
                    Jsonb(decision["output"]),
                    decision.get("score"),
                    decision.get("threshold"),
                    decision.get("flagged"),
                    decision["model_version"],
                ),
            )

    def healthy(self) -> bool:
        try:
            with self.pool.connection(timeout=2) as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def seed(self, df: pl.DataFrame, batch: int = 50_000) -> int:
        """Bulk-load past transactions (history only) with COPY. Skips existing IDs."""
        frame = df.select(list(HISTORY_SCHEMA))
        with self.pool.connection() as conn:
            conn.execute("CREATE TEMP TABLE staging (LIKE transactions) ON COMMIT DROP")
            with conn.cursor().copy(f"COPY staging ({', '.join(COLUMNS)}) FROM STDIN") as copy:
                for chunk in frame.iter_slices(batch):
                    for row in chunk.iter_rows():
                        copy.write_row(row)
            inserted = conn.execute(
                "INSERT INTO transactions SELECT * FROM staging ON CONFLICT DO NOTHING"
            ).rowcount
        return inserted
