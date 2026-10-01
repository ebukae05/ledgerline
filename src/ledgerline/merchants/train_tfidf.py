"""Train the TF-IDF merchant model the API falls back to, and save it.

    python -m ledgerline.merchants.train_tfidf

Fits on the merchant-split training set with the C chosen on validation.
"""

import hashlib
import json
from pathlib import Path

import joblib

from ledgerline.data.load import PROJECT_ROOT
from ledgerline.merchants import tfidf
from ledgerline.merchants.benchmark import TFIDF_C
from ledgerline.merchants.data import load_synthetic, merchant_split

MERCHANT_DIR = PROJECT_ROOT / "artifacts" / "merchants"
MODEL_FILE = "tfidf.joblib"


def main(out_dir: Path = MERCHANT_DIR) -> Path:
    train, _, _ = merchant_split(load_synthetic())
    model = tfidf.fit(train["description"].to_list(), train["category"].to_list(), TFIDF_C)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MODEL_FILE
    joblib.dump(model, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:10]
    meta = {"version": f"tfidf-{digest}", "c": TFIDF_C, "train_rows": train.height}
    (out_dir / "tfidf_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Saved {path.relative_to(PROJECT_ROOT)} ({meta['version']})")
    return path


if __name__ == "__main__":
    main()
