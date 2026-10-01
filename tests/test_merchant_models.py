import json
from types import SimpleNamespace

import pytest

from ledgerline.merchants import rules, tfidf
from ledgerline.merchants.clean import clean, sign
from ledgerline.merchants.llm import GeminiClassifier
from ledgerline.merchants.metrics import classification_report


def test_clean_strips_noise_but_keeps_the_merchant():
    raw = "[debit] PYPL *TRADER JOE'S #552 1234 MAIN ST SEATTLE 98101 WA USA"
    assert clean(raw) == "trader joe's main st seattle wa"
    assert clean("[debit] GEICO EFT PYMT PPD ID: 1978718511") == "geico eft pymt"
    assert clean("[debit] AMAZON MKTPL*K8R2M5VN7") == "amazon mktpl"


def test_sign():
    assert sign("[credit] ACME PAYROLL") == "credit"
    assert sign("no prefix") == "unknown"


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("[debit] OVERDRAFT FEE", "Fees"),
        ("[credit] ACME CORP PAYROLL PPD ID: 1", "Income"),
        ("[debit] Zelle payment to Sam", "Transfer"),
        ("[debit] SQ *BLUE BOTTLE COFFEE", "Restaurants"),
        ("[debit] SOMETHING UNHEARD OF", "Shopping"),
        ("[credit] SOMETHING UNHEARD OF", "Income"),
    ],
)
def test_rules(description, expected):
    assert rules.predict_one(description) == expected


def test_tfidf_learns_and_reports_confidence():
    descriptions = [f"[debit] SAFEWAY #{i}" for i in range(20)] + [
        f"[debit] SHELL OIL {i}" for i in range(20)
    ]
    labels = ["Groceries"] * 20 + ["Transportation"] * 20

    model = tfidf.fit(descriptions, labels)
    predicted, confidence = tfidf.predict_with_confidence(
        model, ["[debit] SAFEWAY #999", "[debit] SHELL OIL 77"]
    )

    assert predicted == ["Groceries", "Transportation"]
    assert all(0.5 < c <= 1 for c in confidence)


def test_classification_report_by_hand():
    r = classification_report(["Rent", "Rent", "Fees", "Fees"], ["Rent", "Fees", "Fees", "Fees"])
    assert r["accuracy"] == 0.75
    assert r["top_confusions"] == [{"true": "Rent", "predicted": "Fees", "count": 1}]
    # Rent: P=1, R=0.5 -> F1 2/3. Fees: P=2/3, R=1 -> F1 0.8. Macro = 0.7333
    assert r["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


class FakeGemini:
    """Stands in for the API: answers by keyword, can drop items, counts calls."""

    def __init__(self, drop_first: bool = False):
        self.calls = 0
        self.drop_first = drop_first
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls += 1
        lines = contents.splitlines()
        items = [
            {"id": i, "category": "Fees" if "FEE" in line else "Groceries"}
            for i, line in enumerate(lines)
        ]
        if self.drop_first and len(lines) > 1:
            items = items[1:]
        usage = SimpleNamespace(
            prompt_token_count=100, candidates_token_count=10 * len(lines), thoughts_token_count=0
        )
        return SimpleNamespace(text=json.dumps(items), usage_metadata=usage)


def test_llm_batches_caches_and_counts_tokens(tmp_path):
    fake = FakeGemini()
    llm = GeminiClassifier(cache_dir=tmp_path, client=fake)

    first = llm.predict(["[debit] ATM FEE", "[debit] SAFEWAY", "[debit] ATM FEE"], batch_size=10)

    assert first == ["Fees", "Groceries", "Fees"]
    assert fake.calls == 1  # duplicates sent once, all in one batch
    assert llm.usage.input_tokens == 100 and llm.usage.output_tokens == 20

    # A new classifier reads the cache from disk: no API calls at all.
    again = GeminiClassifier(cache_dir=tmp_path, client=fake)
    assert again.predict(["[debit] SAFEWAY"]) == ["Groceries"]
    assert fake.calls == 1


def test_llm_retries_items_the_batch_dropped(tmp_path):
    fake = FakeGemini(drop_first=True)
    llm = GeminiClassifier(cache_dir=tmp_path, client=fake)

    assert llm.predict(["[debit] LATE FEE", "[debit] KROGER"]) == ["Fees", "Groceries"]
    assert fake.calls == 2  # one batch, then a single retry for the dropped item


def test_llm_needs_a_known_price(tmp_path):
    with pytest.raises(ValueError, match="no price"):
        GeminiClassifier(model="gemini-made-up", cache_dir=tmp_path, client=FakeGemini())
