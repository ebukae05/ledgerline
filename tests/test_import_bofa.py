import pytest

from ledgerline.merchants.import_bofa import parse_bofa, redact

# Made-up rows in Bank of America's export layout.
EXPORT = """Description,,Summary Amt.
Beginning balance as of 01/01/2026,,"10.00"
Total credits,,"1,250.00"
Total debits,,"-40.00"
Ending balance as of 01/31/2026,,"1,220.00"

Date,Description,Amount,Running Bal.
01/01/2026,Beginning balance as of 01/01/2026,,"10.00"
01/02/2026,"TARGET T-1974 01/02 MOBILE PURCHASE SPRINGFIELD IL","-40.00","-30.00"
01/03/2026,"Zelle payment from JANE Q DOE for "rent share"; Conf# ABC123","1,250.00","1,220.00"
"""


def test_parses_transactions_and_skips_summary_and_balance_rows():
    df = parse_bofa(EXPORT)

    assert df["amount"].to_list() == [-40.0, 1250.0]
    assert df["description"][0] == "TARGET T-1974 01/02 MOBILE PURCHASE SPRINGFIELD IL"


def test_names_are_redacted():
    assert "JANE" not in parse_bofa(EXPORT)["description"][1]
    assert redact('Zelle payment from JANE Q DOE for "rent"; Conf# X1') == (
        'Zelle payment from [NAME] for "rent"; Conf# X1'
    )
    assert redact("Zelle payment to SAM LEE Conf# X2") == "Zelle payment to [NAME] Conf# X2"
    assert redact("ACME DES:PAYROLL INDN:JANE DOE CO ID:123 PPD") == (
        "ACME DES:PAYROLL INDN:[NAME] CO ID:123 PPD"
    )


def test_rejects_files_that_are_not_bofa_exports():
    with pytest.raises(ValueError, match="Bank of America"):
        parse_bofa("Posted Date,Payee,Amount\n")
