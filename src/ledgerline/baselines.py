"""Train the Phase 2 baselines and report validation metrics.

    python -m ledgerline.baselines

Only train and validation are used. The test split stays untouched until the
final Phase 3 comparison.
"""

import json
from pathlib import Path

from ledgerline.data.load import PROJECT_ROOT, load_train
from ledgerline.data.split import time_split
from ledgerline.eval.metrics import report
from ledgerline.models.logreg import fit_logreg, score_logreg
from ledgerline.models.rules import RulesBaseline

REPORT_PATH = PROJECT_ROOT / "reports" / "baselines_val.json"


def run(report_path: Path = REPORT_PATH) -> dict:
    train, val, _ = time_split(load_train())
    y_val = val["isFraud"].to_numpy()

    rules = RulesBaseline().fit(train, val)
    logreg = fit_logreg(train)

    results = {
        "split": "validation",
        "rules": {
            **report(y_val, rules.score(val)),
            "high_amount_cutoff": rules.high_amount,
            "risky_email_domains": rules.risky_domains,
        },
        "logistic_regression": report(y_val, score_logreg(logreg, val)),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> None:
    results = run()
    cols = ["pr_auc", "roc_auc", "recall_at_1pct_fpr", "precision_at_top_1pct"]
    print(f"Validation: {results['rules']['n']:,} rows, {results['rules']['fraud_rate']:.2%} fraud")
    print(f"{'model':<22}" + "".join(f"{c:>24}" for c in cols))
    for name in ("rules", "logistic_regression"):
        print(f"{name:<22}" + "".join(f"{results[name][c]:>24.4f}" for c in cols))
    print(f"Saved to {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
