import polars as pl
import pytest


def make_transactions(rows: list[dict]) -> pl.DataFrame:
    """Build a small transactions frame with every column the models read."""
    defaults = {
        "TransactionID": 0,
        "isFraud": 0,
        "TransactionDT": 86_400,
        "TransactionAmt": 50.0,
        "ProductCD": "W",
        "card4": "visa",
        "card6": "debit",
        "DeviceType": None,
        "P_emaildomain": "gmail.com",
        "id_01": None,
    }
    schema = {
        "TransactionID": pl.Int64,
        "isFraud": pl.Int8,
        "TransactionDT": pl.Int64,
        "TransactionAmt": pl.Float64,
        "ProductCD": pl.String,
        "card4": pl.String,
        "card6": pl.String,
        "DeviceType": pl.String,
        "P_emaildomain": pl.String,
        "id_01": pl.Float32,
    }
    return pl.DataFrame(
        [{**defaults, "TransactionID": i, **row} for i, row in enumerate(rows)], schema=schema
    )


@pytest.fixture
def transactions():
    return make_transactions
