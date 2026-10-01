"""Turn a Bank of America CSV download into a file ready for labeling.

    python -m ledgerline.merchants.import_bofa path/to/stmt.csv

Writes data/private/my_transactions.csv with description, amount and an empty
category column. Person names are replaced with [NAME] (Zelle senders and
recipients, ACH "INDN:" fields) so they never reach the LLM. Refuses to
overwrite a file that already has labels.
"""

import argparse
import csv
import io
import re
from pathlib import Path

import polars as pl

from ledgerline.merchants.data import PRIVATE_PATH

HEADER = "Date,Description,Amount,Running Bal."
NAME_PATTERNS = [
    # "Zelle payment from JANE DOE for ..." / "Zelle payment to JANE DOE Conf# ..."
    (re.compile(r"(Zelle payment (?:from|to) )(.+?)( for | Conf#|;|$)", re.I), r"\1[NAME]\3"),
    # ACH addenda: "INDN:JANE DOE CO ID:..."
    (re.compile(r"(INDN:)(.+?)( CO ID:|$)"), r"\1[NAME]\3"),
]


def redact(description: str) -> str:
    for pattern, replacement in NAME_PATTERNS:
        description = pattern.sub(replacement, description)
    return description


def parse_bofa(text: str) -> pl.DataFrame:
    """Transactions from a BofA export, skipping the summary block and balance rows."""
    start = text.find(HEADER)
    if start == -1:
        raise ValueError(f"no '{HEADER}' header found; is this a Bank of America CSV?")
    rows = []
    for row in csv.DictReader(io.StringIO(text[start:])):
        amount = (row.get("Amount") or "").replace(",", "").strip()
        if not amount:  # "Beginning balance" and blank lines carry no amount
            continue
        rows.append({"description": redact(row["Description"].strip()), "amount": float(amount)})
    return pl.DataFrame(rows, schema={"description": pl.String, "amount": pl.Float64})


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a Bank of America CSV for labeling.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, default=PRIVATE_PATH)
    parser.add_argument("--force", action="store_true", help="overwrite a labeled file")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        existing = pl.read_csv(args.out)
        if existing["category"].drop_nulls().len() > 2:  # more than the template's examples
            raise SystemExit(f"{args.out} already has labels; pass --force to overwrite it.")

    df = parse_bofa(args.source.read_text(encoding="utf-8-sig")).with_columns(
        pl.lit(None, dtype=pl.String).alias("category")
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(args.out)
    redacted = df["description"].str.contains(r"\[NAME\]").sum()
    print(f"Wrote {df.height} transactions to {args.out} ({redacted} with names redacted).")
    print("Open it in Excel and fill in the category column.")


if __name__ == "__main__":
    main()
