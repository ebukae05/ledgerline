"""Train the final fraud model and save it with a version.

    python -m ledgerline.train

Fits on the training split, uses validation to stop adding trees and to set
the flag threshold, and never touches test. The configuration below was
chosen from `python -m ledgerline.experiments` (validation results only).
"""

import subprocess
from datetime import UTC, datetime

from ledgerline.data.load import PROJECT_ROOT, load_train
from ledgerline.data.split import time_split
from ledgerline.eval.metrics import report, threshold_at_fpr
from ledgerline.features.pipeline import add_features, feature_groups
from ledgerline.models.lgbm import PARAMS, train_lgbm
from ledgerline.models.registry import save_model

FEATURE_GROUPS = ["basic", "history", "raw"]
BALANCE_CLASSES = True
MAX_FPR = 0.01


def _git_commit() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=PROJECT_ROOT
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")


def main() -> None:
    df = add_features(load_train())
    groups = feature_groups(df)
    features = [f for g in FEATURE_GROUPS for f in groups[g]]
    train, val, _ = time_split(df)

    model = train_lgbm(train, val, features, balance_classes=BALANCE_CLASSES)
    y_val = val["isFraud"].to_numpy()
    val_scores = model.score(val)
    threshold = threshold_at_fpr(y_val, val_scores, MAX_FPR)

    folder = save_model(
        model,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "feature_groups": FEATURE_GROUPS,
            "balance_classes": BALANCE_CLASSES,
            "params": PARAMS,
            "trees": model.booster.best_iteration,
            "threshold": threshold,
            "threshold_rule": f"lowest score with validation FPR <= {MAX_FPR:.0%}",
            "train_days": [int(train["day"].min()), int(train["day"].max())],
            "val_days": [int(val["day"].min()), int(val["day"].max())],
            "train_rows": train.height,
            "val_metrics": report(y_val, val_scores),
        },
    )
    print(f"Saved {folder.relative_to(PROJECT_ROOT)}")
    print(f"Threshold {threshold:.4f} (validation FPR <= {MAX_FPR:.0%})")


if __name__ == "__main__":
    main()
