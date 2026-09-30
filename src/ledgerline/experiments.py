"""Phase 3 model comparison and ablation, on the validation split only.

    python -m ledgerline.experiments

Every run here is judged on validation. The test split is used once, by
ledgerline.evaluate, after the final configuration is chosen.

LightGBM results move a little with the random seed (row and feature
sampling), so each variant runs with several seeds and reports mean and
spread. A difference smaller than that spread is noise, not a finding.
Results are saved after every run, so an interrupted run keeps its progress.
"""

import json
import statistics

from ledgerline.data.load import PROJECT_ROOT, load_train
from ledgerline.data.split import time_split
from ledgerline.eval.metrics import report
from ledgerline.features.history import HISTORY_FEATURES
from ledgerline.features.pipeline import add_features, feature_groups
from ledgerline.models.lgbm import train_lgbm
from ledgerline.models.logreg import fit_logreg, score_logreg
from ledgerline.models.rules import RulesBaseline

REPORT_PATH = PROJECT_ROOT / "reports" / "experiments_val.json"
SEEDS = (42, 7, 123)
TOP_IMPORTANCE = 20
METRICS = ("pr_auc", "roc_auc", "recall_at_1pct_fpr", "precision_at_top_1pct")

# History feature subgroups to remove one at a time.
ABLATIONS = {
    "velocity (1h/24h/7d counts)": ["_n_1h", "_n_24h", "_n_7d"],
    "prior count": ["_n_prior"],
    "time since previous": ["_secs_since_prev"],
    "amount vs card mean": ["_amt_vs_mean"],
    "device/email seen before": ["_device_seen_before", "_email_seen_before"],
    "card1 key (all card_*)": ["card_"],
    "uid key (all uid_*)": ["uid_"],
    "all history features": ["card_", "uid_"],
}


def _without(features: list[str], patterns: list[str]) -> list[str]:
    dropped = {f for f in HISTORY_FEATURES if any(p in f for p in patterns)}
    return [f for f in features if f not in dropped]


def _summarize(runs: list[dict]) -> dict:
    out = {}
    for m in METRICS:
        values = [r[m] for r in runs]
        out[m] = statistics.mean(values)
        out[f"{m}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    out["runs"] = runs
    return out


def run() -> dict:
    df = add_features(load_train())
    groups = feature_groups(df)
    train, val, _ = time_split(df)
    y_val = val["isFraud"].to_numpy()
    basic, history, raw = groups["basic"], groups["history"], groups["raw"]

    results: dict = {"split": "validation", "seeds": list(SEEDS), "models": {}, "ablation": {}}

    def save():
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(results, indent=2) + "\n")

    def log(name, r):
        spread = f" ± {r['pr_auc_std']:.4f}" if r.get("pr_auc_std") else ""
        print(f"{name:<45} PR-AUC {r['pr_auc']:.4f}{spread}", flush=True)

    def lgbm_runs(features, balanced, seeds):
        runs = []
        for seed in seeds:
            m = train_lgbm(train, val, features, balance_classes=balanced, seed=seed)
            runs.append(
                {**report(y_val, m.score(val)), "seed": seed, "trees": m.booster.best_iteration}
            )
        return _summarize(runs), m

    def importance(model):
        gain = model.booster.feature_importance("gain")
        ranked = sorted(zip(model.features, gain, strict=True), key=lambda x: -x[1])
        return {
            "top": [
                {"feature": f, "gain_share": float(g / gain.sum())}
                for f, g in ranked[:TOP_IMPORTANCE]
            ],
            "history_share": float(sum(g for f, g in ranked if f in HISTORY_FEATURES) / gain.sum()),
        }

    for name, scores in [
        ("rules", RulesBaseline().fit(train, val).score(val)),
        ("logistic_regression", score_logreg(fit_logreg(train), val)),
    ]:
        results["models"][name] = report(y_val, scores)
        log(name, results["models"][name])
        save()

    # (features, class weighting, seeds). A full-feature fit takes ~7 minutes,
    # so those get fewer seeds; the fast variants get all of them.
    full = basic + history + raw
    variants = {
        "lgbm_basic": (basic, False, SEEDS),
        "lgbm_basic+history": (basic + history, False, SEEDS),
        "lgbm_basic+raw": (basic + raw, False, SEEDS[:2]),
        "lgbm_basic+history+raw": (full, False, SEEDS[:2]),
        "lgbm_basic+history+raw_balanced": (full, True, SEEDS[:1]),
    }
    results["importance"] = {}
    for name, (features, balanced, seeds) in variants.items():
        summary, model = lgbm_runs(features, balanced, seeds)
        results["models"][name] = {**summary, "n_features": len(features), "balanced": balanced}
        if name in ("lgbm_basic+history", "lgbm_basic+history+raw"):
            results["importance"][name] = importance(model)
        log(name, summary)
        save()

    # Which history feature matters? Ablate on basic+history, where history is
    # the main signal and a fit takes seconds, so every ablation gets all seeds.
    reference_name = "lgbm_basic+history"
    reference = results["models"][reference_name]
    results["ablation_reference"] = reference_name
    for name, patterns in ABLATIONS.items():
        features = _without(basic + history, patterns)
        summary, _ = lgbm_runs(features, False, SEEDS)
        summary["features_removed"] = len(basic + history) - len(features)
        summary["pr_auc_change"] = summary["pr_auc"] - reference["pr_auc"]
        results["ablation"][name] = summary
        log(f"  without {name}", summary)
        save()

    return results


if __name__ == "__main__":
    run()
    print(f"Saved to {REPORT_PATH.relative_to(PROJECT_ROOT)}")
