"""Load bank descriptors and split them by merchant.

The synthetic set has ~68k rows built from a few hundred merchant names, and a
third of rows are exact duplicates. A random split would put "PP*SAFEWAY 1234"
in train and "PP*SAFEWAY 5678" in test, so a model could score well just by
memorizing merchant names. Splitting by merchant means every test merchant is
new to the model, which is the situation on a real bank statement.

The merchant isn't a column, so it's estimated: a row's merchant key is its
first word that belongs to at most two categories. Boilerplate words
("payment", "withdrawal", "paypal") span many categories and get skipped;
brand words ("geico", "target") don't. This uses labels only to decide which
rows travel together, never as model input.
"""

import hashlib
from pathlib import Path
from typing import NamedTuple

import polars as pl

from ledgerline.data.load import PROJECT_ROOT

SYNTHETIC_PATH = PROJECT_ROOT / "data" / "raw" / "merchants" / "transactions-synthetic.csv"
PRIVATE_PATH = PROJECT_ROOT / "data" / "private" / "my_transactions.csv"

CATEGORIES = [
    "Education",
    "Entertainment",
    "Fees",
    "Groceries",
    "Healthcare",
    "Income",
    "Insurance",
    "Mortgage",
    "Personal Care",
    "Rent",
    "Restaurants",
    "Shopping",
    "Subscription",
    "Transfer",
    "Transportation",
    "Travel",
    "Utilities",
]

BRAND_MAX_CATEGORIES = 2
SPLIT_SALT = "ledgerline-merchants-v1"
TRAIN_PCT, VAL_PCT = 70, 15


class Splits(NamedTuple):
    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame


def load_synthetic(path: Path = SYNTHETIC_PATH) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download transactions-synthetic.csv from "
            "huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2"
        )
    df = pl.read_csv(path)
    unknown = set(df["category"].unique()) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"unexpected categories in {path.name}: {sorted(unknown)}")
    return df.select("description", "category")


def load_private(path: Path = PRIVATE_PATH) -> pl.DataFrame:
    """Your own hand-labeled transactions: description, amount, category.

    Bank exports carry the sign in the amount column, while the synthetic set
    puts it in the text, so the [debit]/[credit] prefix is added here.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy templates/my_transactions_template.csv there, "
            "replace the example rows with your transactions, and label each one."
        )
    df = pl.read_csv(path).filter(pl.col("category").is_not_null())
    unknown = set(df["category"].unique()) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"categories not in the list of 17: {sorted(unknown)}")
    sign = pl.when(pl.col("amount") < 0).then(pl.lit("[debit] ")).otherwise(pl.lit("[credit] "))
    return df.select(
        pl.concat_str([sign, pl.col("description").str.strip_chars()]).alias("description"),
        "category",
    )


def _body() -> pl.Expr:
    return pl.col("description").str.to_lowercase().str.replace(r"^\[(debit|credit)\]\s*", "")


def merchant_keys(df: pl.DataFrame) -> pl.Series:
    """Estimated merchant for each row (see module docstring)."""
    rows = df.with_row_index("_i").with_columns(
        _body().str.extract_all(r"[a-z][a-z&']+").alias("_tok")
    )
    tokens = (
        rows.select("_i", "category", "_tok").explode("_tok", empty_as_null=True).drop_nulls("_tok")
    )
    tokens = tokens.with_columns(pl.int_range(pl.len()).over("_i").alias("_pos"))
    spread = tokens.group_by("_tok").agg(pl.col("category").n_unique().alias("_spread"))
    brand = (
        tokens.join(spread, on="_tok")
        .filter(pl.col("_spread") <= BRAND_MAX_CATEGORIES)
        .sort("_i", "_pos")
        .group_by("_i")
        .agg(pl.col("_tok").first().alias("_brand"))
    )
    letters_only = (
        _body().str.replace_all(r"[^a-z ]", " ").str.replace_all(r"\s+", " ").str.strip_chars()
    )
    return (
        rows.join(brand, on="_i", how="left")
        .sort("_i")
        .select(pl.coalesce("_brand", letters_only.alias("_l")).alias("merchant_key"))
        .to_series()
    )


def _bucket(key: str) -> int:
    """Stable 0-99 bucket for a merchant key, the same on every machine and run."""
    return int(hashlib.sha256(f"{SPLIT_SALT}:{key}".encode()).hexdigest(), 16) % 100


def merchant_split(df: pl.DataFrame) -> Splits:
    """~70/15/15 split where every merchant lands in exactly one split."""
    keyed = df.with_columns(merchant_keys(df)).with_columns(
        pl.col("merchant_key").map_elements(_bucket, return_dtype=pl.Int64).alias("_bucket")
    )
    b = pl.col("_bucket")
    return Splits(
        train=keyed.filter(b < TRAIN_PCT).drop("_bucket"),
        val=keyed.filter((b >= TRAIN_PCT) & (b < TRAIN_PCT + VAL_PCT)).drop("_bucket"),
        test=keyed.filter(b >= TRAIN_PCT + VAL_PCT).drop("_bucket"),
    )
