"""Score the saved model and the baselines on the test split, once.

    python -m ledgerline.evaluate

Test data is the stand-in for "the future". Looking at it, changing something,
and looking again turns it into a second validation set and makes the reported
numbers optimistic. So this refuses to run twice unless you pass --force, and
the report records which model version it evaluated.
"""

import argparse
import json
from datetime import UTC, datetime

import numpy as np

from ledgerline.data.load import PROJECT_ROOT, load_train
from ledgerline.data.split import time_split
from ledgerline.eval.metrics import report
from ledgerline.features.pipeline import add_features
from ledgerline.models.logreg import fit_logreg, score_logreg
from ledgerline.models.registry import load_model
from ledgerline.models.rules import RulesBaseline

REPORT_PATH = PROJECT_ROOT / "reports" / "test_results.json"


def operating_point(y: np.ndarray, scores: np.ndarray, amounts: np.ndarray, threshold: float):
    """What happens on test when flagging at the threshold chosen on validation."""
    flagged = scores >= threshold
    fraud = y == 1
    return {
        "threshold": threshold,
        "flag_rate": float(flagged.mean()),
        "fpr": float(flagged[~fraud].mean()),
        "recall": float(flagged[fraud].mean()),
        "precision": float(y[flagged].mean()) if flagged.any() else 0.0,
        "fraud_dollars_caught_share": float(amounts[flagged & fraud].sum() / amounts[fraud].sum()),
        "fraud_dollars_per_1000_flags": float(
            amounts[flagged & fraud].sum() / max(flagged.sum(), 1) * 1000
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="overwrite an existing test report")
    args = parser.parse_args()

    if REPORT_PATH.exists() and not args.force:
        existing = json.loads(REPORT_PATH.read_text())
        raise SystemExit(
            f"Test set already evaluated for {existing['model_version']} "
            f"on {existing['evaluated_at']}. See {REPORT_PATH.relative_to(PROJECT_ROOT)}. "
            "Re-running after changes turns test into validation; pass --force only if you mean it."
        )

    model, meta = load_model()
    df = add_features(load_train())
    train, val, test = time_split(df)
    y = test["isFraud"].to_numpy()

    scores = model.score(test)
    results = {
        "model_version": meta["version"],
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": "test",
        "test_days": [int(test["day"].min()), int(test["day"].max())],
        "models": {
            "rules": report(y, RulesBaseline().fit(train, val).score(test)),
            "logistic_regression": report(y, score_logreg(fit_logreg(train), test)),
            "lgbm": report(y, scores),
        },
        "operating_point": operating_point(
            y, scores, test["TransactionAmt"].to_numpy(), meta["threshold"]
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
