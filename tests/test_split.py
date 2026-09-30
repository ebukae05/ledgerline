import polars as pl
import pytest

from ledgerline.data.split import time_split


def make_df(times):
    return pl.DataFrame({"TransactionID": range(len(times)), "TransactionDT": times})


def test_no_later_split_contains_an_earlier_time():
    # Shuffled input: the split must not rely on rows arriving in time order.
    df = make_df([5, 1, 9, 3, 7, 2, 8, 0, 6, 4] * 10).sample(fraction=1.0, shuffle=True, seed=0)

    train, val, test = time_split(df)

    assert train["TransactionDT"].max() < val["TransactionDT"].min()
    assert val["TransactionDT"].max() < test["TransactionDT"].min()


def test_every_row_lands_in_exactly_one_split():
    df = make_df(list(range(1000)))

    train, val, test = time_split(df)

    ids = pl.concat([train, val, test])["TransactionID"]
    assert ids.len() == 1000
    assert ids.n_unique() == 1000


def test_fractions_are_close_to_requested():
    df = make_df(list(range(1000)))

    train, val, test = time_split(df)

    assert (len(train), len(val), len(test)) == (700, 150, 150)


def test_rows_sharing_a_timestamp_stay_together_at_the_boundary():
    # 10 rows at t=0..5 and t=6 repeated 5 times: the 70% cutoff falls inside
    # the run of 6s, which must all move to validation together.
    df = make_df([0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 7, 8, 9])

    train, val, test = time_split(df)

    assert 6 not in train["TransactionDT"].to_list()
    assert val["TransactionDT"].to_list().count(6) == 5
    assert train["TransactionDT"].max() < val["TransactionDT"].min()


def test_each_split_is_sorted_by_time():
    df = make_df([3, 1, 2, 0, 9, 8, 7, 6, 5, 4] * 3)

    for part in time_split(df):
        assert part["TransactionDT"].is_sorted()


@pytest.mark.parametrize(
    ("train_frac", "val_frac"), [(0.0, 0.2), (0.7, 0.0), (0.8, 0.2), (1.2, 0.1)]
)
def test_rejects_bad_fractions(train_frac, val_frac):
    with pytest.raises(ValueError, match="train_frac"):
        time_split(make_df(list(range(100))), train_frac, val_frac)


def test_rejects_missing_time_column():
    with pytest.raises(ValueError, match="not in DataFrame"):
        time_split(make_df([1, 2, 3]), time_col="nope")


def test_rejects_null_times():
    with pytest.raises(ValueError, match="nulls"):
        time_split(make_df([1, None, 3]))


def test_rejects_split_that_would_be_empty():
    with pytest.raises(ValueError, match="empty"):
        time_split(make_df([1, 1, 1, 1, 2]))
