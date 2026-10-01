import polars as pl
import pytest

from ledgerline.merchants.data import load_private, merchant_keys, merchant_split


def frame(rows):
    return pl.DataFrame(rows, schema=["description", "category"], orient="row")


def test_merchant_key_skips_boilerplate_and_finds_the_brand():
    df = frame(
        [
            ("[debit] Payment to GEICO", "Insurance"),
            ("[debit] GEICO EFT PYMT PPD ID: 1978718511", "Insurance"),
            ("[debit] PAYPAL *SAFEWAY 123", "Groceries"),
            ("[debit] Payment to SAFEWAY", "Groceries"),
            ("[debit] PAYPAL *NETFLIX", "Subscription"),
            ("[debit] Payment to NETFLIX", "Subscription"),
            ("[debit] PAYPAL *UBER TRIP", "Transportation"),
            ("[debit] Payment to UBER", "Transportation"),
        ]
    )

    keys = merchant_keys(df).to_list()

    # "payment" and "paypal" span 4 categories, so they're skipped.
    assert keys == ["geico", "geico", "safeway", "safeway", "netflix", "netflix", "uber", "uber"]


def test_split_never_puts_a_merchant_in_two_splits():
    rows = [
        (f"[debit] MERCHANT{chr(97 + i % 26)}{chr(97 + i // 26)} #{n}", "Shopping")
        for i in range(300)
        for n in range(3)
    ]
    df = frame(rows)

    train, val, test = merchant_split(df)

    sets = [set(s["merchant_key"]) for s in (train, val, test)]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert train.height + val.height + test.height == df.height
    assert 0.55 < train.height / df.height < 0.85


def test_split_is_the_same_every_run():
    df = frame([(f"[debit] STORE{i} #1", "Shopping") for i in range(200)])

    first, second = merchant_split(df), merchant_split(df)

    assert all(a.equals(b) for a, b in zip(first, second, strict=True))


def test_private_set_gets_sign_prefix_and_rejects_unknown_categories(tmp_path):
    path = tmp_path / "mine.csv"
    path.write_text(
        "description,amount,category\n"
        "  TRADER JOE'S #552 ,-41.20,Groceries\n"
        "ACME PAYROLL,2100.00,Income\n"
        "UNLABELED THING,-5.00,\n"
    )

    df = load_private(path)

    assert df["description"].to_list() == ["[debit] TRADER JOE'S #552", "[credit] ACME PAYROLL"]

    path.write_text("description,amount,category\nX,-1,Snacks\n")
    with pytest.raises(ValueError, match="Snacks"):
        load_private(path)
