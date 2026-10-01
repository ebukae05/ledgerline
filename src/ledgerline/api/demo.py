"""Walk through the API end to end, for a live demo or a screen recording.

    python -m ledgerline.api.demo                       # API at localhost:8000
    python -m ledgerline.api.demo --url http://host:8000

Uses real test-split transactions (data the model never trained on), so it
needs the Kaggle data locally. Shows: health, a normal transaction, a real
fraud flagged with reasons, a bad request, a batch with one bad record,
merchant categorization, and the audit rows those calls wrote.
"""

import argparse
import json
import os
import time

import httpx
import numpy as np
import polars as pl

from ledgerline.data.load import load_train
from ledgerline.data.split import time_split

DEFAULT_DB = "postgresql://ledgerline:ledgerline@localhost:5432/ledgerline"
MERCHANT_EXAMPLES = [
    ("PAYPAL *DATACAMP JYF7455M6J", "debit"),
    ("TST* BLUE BOTTLE COFFEE OAKLAND CA", "debit"),
    ("ACME CORP PAYROLL PPD ID: 4455667788", "credit"),
]
# Columns the API accepts; anything else in the dataset (labels, derived) is dropped.
DROP = {"isFraud", "day"}


def step(title: str) -> None:
    print(f"\n\033[1m== {title} ==\033[0m", flush=True)
    time.sleep(0.4)


def show(response: httpx.Response, keep: tuple[str, ...] | None = None) -> dict:
    body = response.json()
    shown = {k: body[k] for k in keep if k in body} if keep else body
    print(f"HTTP {response.status_code}")
    print(json.dumps(shown, indent=2, default=str))
    return body


def payload(row: dict) -> dict:
    return {
        k: v
        for k, v in row.items()
        if k not in DROP and v is not None and not (isinstance(v, float) and np.isnan(v))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo the Ledgerline API.")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    api = httpx.Client(base_url=args.url, timeout=30)
    try:
        api.get("/health")
    except httpx.ConnectError:
        raise SystemExit(f"No API at {args.url}. Start it with `docker compose up -d`.") from None

    _, _, test = time_split(load_train())
    legit = test.filter(pl.col("isFraud") == 0).sample(50, seed=1)
    frauds = test.filter(pl.col("isFraud") == 1).sample(50, seed=1)
    score_keys = ("transaction_id", "score", "flagged", "threshold", "model_version")

    step("1. Health: model loaded, database reachable")
    show(api.get("/health"))

    step("2. A normal transaction from the test set (never seen in training)")
    show(api.post("/score", json=payload(legit.row(0, named=True))), score_keys)

    step("3. A real fraud from the test set: flagged, with reasons")
    for row in frauds.iter_rows(named=True):
        response = api.post("/score", json=payload(row))
        if response.json().get("flagged"):
            body = show(response, score_keys)
            print("Reasons:")
            for reason in body["reasons"]:
                print(f"  +{reason['contribution']:.2f}  {reason['text']}")
            break

    step("4. Bad input: negative amount and a misspelled field -> 422, never a 500")
    bad = {**payload(legit.row(1, named=True)), "TransactionAmt": -25.0, "Amout": 10}
    show(api.post("/score", json=bad))

    step("5. Batch of 3 where one record is broken: the other two still score")
    batch = [payload(legit.row(i, named=True)) for i in (2, 3, 4)]
    batch[1]["ProductCD"] = "Z"
    body = api.post("/score/batch", json={"transactions": batch}).json()
    print(f"scored {body['scored']}, failed {body['failed']}")
    print(json.dumps(body["errors"], indent=2))

    step("6. Merchant categorization from raw bank descriptors")
    for description, direction in MERCHANT_EXAMPLES:
        r = api.post("/merchant", json={"description": description, "direction": direction})
        out = r.json()
        conf = f", confidence {out['confidence']:.2f}" if out["confidence"] is not None else ""
        print(f"{description:<40} -> {out['category']} (via {out['method']}{conf})")

    step("7. Audit trail: every decision above is in Postgres")
    try:
        import psycopg

        with psycopg.connect(os.environ.get("DATABASE_URL", DEFAULT_DB)) as conn:
            rows = conn.execute(
                "SELECT created_at, endpoint, transaction_id, round(score::numeric, 4), "
                "flagged, model_version, left(input_hash, 12) FROM decisions "
                "ORDER BY id DESC LIMIT 8"
            ).fetchall()
        print(
            f"{'time':<10} {'endpoint':<10} {'txn':>9} {'score':>7} {'flag':<5} {'model':<17} hash"
        )
        for at, endpoint, txn, score, flagged, version, digest in rows:
            score_text = "" if score is None else str(score)
            flag_text = "" if flagged is None else str(flagged)
            print(
                f"{at:%H:%M:%S}  {endpoint:<10} {txn or '':>9} {score_text:>7} "
                f"{flag_text:<5} {version:<17} {digest}"
            )
    except Exception as error:  # demo still works without direct DB access
        print(f"(couldn't read the audit table: {error})")


if __name__ == "__main__":
    main()
