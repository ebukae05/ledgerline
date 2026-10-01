"""Load past transactions into the API's history table.

    python -m ledgerline.api.seed

Copies the history columns of every training-data transaction into Postgres,
so the API sees the same card history the model was evaluated with. Safe to
run twice: existing transaction IDs are skipped.
"""

import os
import time

from ledgerline.api.store import PostgresStore
from ledgerline.data.load import load_train


def main() -> None:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set.")
    start = time.perf_counter()
    inserted = PostgresStore(url).seed(load_train())
    print(f"Inserted {inserted:,} transactions in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
