"""Load the IEEE-CIS training data into one Polars DataFrame.

The raw CSVs are ~700 MB and slow to parse, so the joined result is cached as
Parquet in data/processed/. Delete the cache (or pass refresh=True) to rebuild.
"""

import os
from pathlib import Path

import polars as pl

# Repo root when running from source; set PROJECT_ROOT when the package is
# installed elsewhere (e.g. in the Docker image).
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[3])
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"
CACHE_FILE = "train.parquet"

SECONDS_PER_DAY = 86_400

# Money stays Float64 so cents are exact after the round trip; everything else
# numeric is shrunk to the smallest type that holds its values.
KEEP_FLOAT64 = {"TransactionAmt"}


def _read_csv(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the IEEE-CIS data from Kaggle and put "
            f"{TRANSACTION_FILE} and {IDENTITY_FILE} in {path.parent}."
        )
    # Scan every row to infer types: a column that looks like ints for the
    # first 10k rows can still hold floats later on.
    return pl.read_csv(path, infer_schema_length=None)


def _downcast(df: pl.DataFrame) -> pl.DataFrame:
    columns = []
    for name, dtype in df.schema.items():
        if name in KEEP_FLOAT64:
            continue
        if dtype == pl.Float64:
            columns.append(df[name].cast(pl.Float32))
        elif dtype.is_integer():
            # Series.shrink_dtype looks at the actual values; the Expr version is a no-op.
            columns.append(df[name].shrink_dtype())
    return df.with_columns(columns)


def load_raw(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    """Read both CSVs and left-join identity onto transactions.

    Left join because only some transactions have an identity row; the rest
    keep nulls in the identity columns rather than being dropped.
    """
    transactions = _read_csv(raw_dir / TRANSACTION_FILE)
    identity = _read_csv(raw_dir / IDENTITY_FILE)

    df = transactions.join(identity, on="TransactionID", how="left", validate="1:1")
    df = df.with_columns((pl.col("TransactionDT") // SECONDS_PER_DAY).alias("day"))
    return _downcast(df)


def load_train(
    raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR, refresh: bool = False
) -> pl.DataFrame:
    """Return the joined training data, reading the Parquet cache when present."""
    cache = processed_dir / CACHE_FILE
    if cache.exists() and not refresh:
        return pl.read_parquet(cache)

    df = load_raw(raw_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache)
    return df
