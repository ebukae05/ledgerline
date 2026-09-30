import math

from ledgerline.features.basic import CATEGORICAL, NUMERIC, basic_features


def test_one_output_row_per_input_row_with_expected_columns(transactions):
    df = transactions([{}, {}, {}])

    features = basic_features(df)

    assert features.height == 3
    assert set(features.columns) == set(CATEGORICAL + NUMERIC)


def test_values(transactions):
    df = transactions(
        [
            {"TransactionAmt": 31.953, "TransactionDT": 86_400 + 7 * 3_600, "id_01": -5.0},
            {"TransactionAmt": 10.0, "DeviceType": None, "P_emaildomain": None},
        ]
    )

    first, second = basic_features(df).to_dicts()

    assert first["hour"] == "7"
    assert first["log_amount"] == math.log1p(31.953)
    assert first["odd_cents"] == 1
    assert first["has_identity"] == 1
    assert second["odd_cents"] == 0
    assert second["has_identity"] == 0
    assert second["DeviceType"] == "missing"
    assert second["P_emaildomain"] == "missing"


def test_a_row_features_do_not_depend_on_other_rows(transactions):
    # Row-level features must be identical whether computed alone or in a batch,
    # which is what lets the API score one transaction at a time.
    batch = transactions([{"TransactionAmt": 12.5}, {"TransactionAmt": 900.0, "card6": "credit"}])

    alone = basic_features(batch.slice(1, 1))

    assert basic_features(batch).slice(1, 1).equals(alone)
