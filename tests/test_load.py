import polars as pl
import pytest

from ledgerline.data.load import load_raw, load_train


@pytest.fixture
def raw_dir(tmp_path):
    (tmp_path / "train_transaction.csv").write_text(
        "TransactionID,isFraud,TransactionDT,TransactionAmt,card1,V1\n"
        "1,0,86400,10.25,1000,1.5\n"
        "2,1,90000,99.99,1001,\n"
        "3,0,172800,5.00,1000,0.5\n"
    )
    (tmp_path / "train_identity.csv").write_text("TransactionID,id_01,DeviceType\n2,-5.0,mobile\n")
    return tmp_path


def test_left_join_keeps_transactions_without_identity(raw_dir):
    df = load_raw(raw_dir)

    assert df.height == 3
    assert df.filter(pl.col("TransactionID") == 2)["DeviceType"].item() == "mobile"
    assert df.filter(pl.col("TransactionID") == 1)["DeviceType"].item() is None


def test_day_is_whole_days_since_reference(raw_dir):
    df = load_raw(raw_dir).sort("TransactionID")

    assert df["day"].to_list() == [1, 1, 2]


def test_amount_keeps_full_precision_and_other_floats_are_downcast(raw_dir):
    df = load_raw(raw_dir)

    assert df.schema["TransactionAmt"] == pl.Float64
    assert df.schema["V1"] == pl.Float32
    assert df.schema["id_01"] == pl.Float32
    assert df.schema["card1"] in (pl.Int16, pl.Int32)


def test_missing_file_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Download the IEEE-CIS data"):
        load_raw(tmp_path)


def test_cache_is_written_then_reused(raw_dir, tmp_path):
    processed = tmp_path / "processed"

    first = load_train(raw_dir, processed)
    (raw_dir / "train_transaction.csv").unlink()  # prove the second call never reads CSVs
    second = load_train(raw_dir, processed)

    assert (processed / "train.parquet").exists()
    assert first.equals(second)
