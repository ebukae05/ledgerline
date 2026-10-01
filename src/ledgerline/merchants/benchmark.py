"""Compare rules, TF-IDF and Gemini on bank descriptors.

    python -m ledgerline.merchants.benchmark          # validation: tune and compare
    python -m ledgerline.merchants.benchmark --test   # synthetic test set, once
    python -m ledgerline.merchants.benchmark --real   # your hand-labeled set, once

Validation scores rules and TF-IDF on every row, and all approaches (plus the
cascade) on a fixed 1,000-row sample so the LLM bill stays small. The test run
and real runs use only settings chosen on validation (TFIDF_C,
CASCADE_THRESHOLD), and each refuses to run twice without --force.

Cascade: TF-IDF answers when it's confident; only the rest go to the LLM.
"""

import argparse
import json
import random
import statistics
import time
from datetime import UTC, datetime

import numpy as np

from ledgerline.data.load import PROJECT_ROOT
from ledgerline.merchants import rules, tfidf
from ledgerline.merchants.data import load_private, load_synthetic, merchant_split
from ledgerline.merchants.llm import GeminiClassifier
from ledgerline.merchants.metrics import classification_report

VAL_REPORT = PROJECT_ROOT / "reports" / "merchants_val.json"
TEST_REPORT = PROJECT_ROOT / "reports" / "merchants_test.json"
REAL_REPORT = PROJECT_ROOT / "reports" / "merchants_real.json"
VAL_SAMPLE = 1_000
LATENCY_SAMPLE = 200
LLM_LATENCY_SAMPLE = 20
SEED = 42
CASCADE_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)

# Chosen on validation (best macro-F1); see reports/merchants_val.json.
TFIDF_C = 3.0
CASCADE_THRESHOLD = 0.95


def _p50_ms(fn, items) -> float:
    times = []
    for item in items:
        start = time.perf_counter()
        fn(item)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def _score(truth, predicted, **extra) -> dict:
    report = classification_report(truth, [p or "Unanswered" for p in predicted])
    return {**report, **extra}


def _cascade(tfidf_pred, tfidf_conf, llm_pred, threshold):
    use_llm = tfidf_conf < threshold
    merged = [
        from_llm if sent else own
        for own, from_llm, sent in zip(tfidf_pred, llm_pred, use_llm, strict=True)
    ]
    return merged, float(use_llm.mean())


def evaluate_set(name, df, model, llm, threshold, latency=False) -> dict:
    """All approaches on one labeled set."""
    texts, truth = df["description"].to_list(), df["category"].to_list()
    t_pred, t_conf = tfidf.predict_with_confidence(model, texts)
    llm_before = (llm.usage.input_tokens, llm.usage.output_tokens)
    l_pred = llm.predict(texts)
    tokens_in = llm.usage.input_tokens - llm_before[0]
    tokens_out = llm.usage.output_tokens - llm_before[1]
    llm_cost_per_1k = None
    if tokens_in:  # only measurable when answers weren't already cached
        price_in, price_out = llm.cost_rates
        llm_cost_per_1k = (tokens_in * price_in + tokens_out * price_out) / 1e6 / len(texts) * 1000

    c_pred, llm_share = _cascade(t_pred, np.asarray(t_conf), l_pred, threshold)
    print(f"{name}: {len(texts):,} rows", flush=True)
    out = {
        "rules": _score(truth, rules.predict(texts)),
        "tfidf": _score(truth, t_pred),
        "llm": _score(truth, l_pred, cost_per_1k_usd=llm_cost_per_1k),
        "cascade": _score(
            truth,
            c_pred,
            threshold=threshold,
            share_sent_to_llm=llm_share,
            cost_per_1k_usd=llm_cost_per_1k * llm_share if llm_cost_per_1k else None,
        ),
    }
    if latency:
        sample = random.Random(SEED).sample(texts, min(LATENCY_SAMPLE, len(texts)))
        out["rules"]["latency_p50_ms"] = _p50_ms(rules.predict_one, sample)
        out["tfidf"]["latency_p50_ms"] = _p50_ms(lambda t: model.predict([t]), sample)
        llm_times = llm.time_single_calls(sample[:LLM_LATENCY_SAMPLE])
        out["llm"]["latency_p50_ms"] = statistics.median(llm_times) * 1000
    for approach, r in out.items():
        print(
            f"  {approach:<8} accuracy {r['accuracy']:.3f}  macro-F1 {r['macro_f1']:.3f}",
            flush=True,
        )
    return out


def run_validation() -> dict:
    train, val, _ = merchant_split(load_synthetic())
    texts, truth = val["description"].to_list(), val["category"].to_list()
    results: dict = {"split": "validation", "tfidf_c_grid": {}, "cascade_grid": {}}

    for c in tfidf.C_GRID:
        m = tfidf.fit(train["description"].to_list(), train["category"].to_list(), c)
        results["tfidf_c_grid"][str(c)] = classification_report(truth, m.predict(texts))["macro_f1"]
    best_c = max(tfidf.C_GRID, key=lambda c: results["tfidf_c_grid"][str(c)])
    model = tfidf.fit(train["description"].to_list(), train["category"].to_list(), best_c)
    results["tfidf_c"] = best_c
    results["full_val"] = {
        "rules": classification_report(truth, rules.predict(texts)),
        "tfidf": classification_report(truth, model.predict(texts)),
    }

    sample = val.sample(VAL_SAMPLE, seed=SEED)
    llm = GeminiClassifier()
    results["llm_model"] = llm.model
    results["sample"] = evaluate_set("val sample", sample, model, llm, CASCADE_THRESHOLD)

    s_texts, s_truth = sample["description"].to_list(), sample["category"].to_list()
    t_pred, t_conf = tfidf.predict_with_confidence(model, s_texts)
    l_pred = llm.predict(s_texts)  # cached
    for threshold in CASCADE_GRID:
        merged, share = _cascade(t_pred, np.asarray(t_conf), l_pred, threshold)
        r = classification_report(s_truth, [p or "Unanswered" for p in merged])
        results["cascade_grid"][str(threshold)] = {
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "share_sent_to_llm": share,
        }
    VAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    VAL_REPORT.write_text(json.dumps(results, indent=2) + "\n")
    return results


def _run_once(report_path, name, df, latency: bool, force: bool) -> dict:
    if report_path.exists() and not force:
        raise SystemExit(
            f"{name} already evaluated ({report_path.relative_to(PROJECT_ROOT)}). "
            "Pass --force only if you mean to re-use it."
        )
    train, _, _ = merchant_split(load_synthetic())
    model = tfidf.fit(train["description"].to_list(), train["category"].to_list(), TFIDF_C)
    llm = GeminiClassifier()
    results = {
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "llm_model": llm.model,
        "tfidf_c": TFIDF_C,
        "cascade_threshold": CASCADE_THRESHOLD,
        **evaluate_set(name, df, model, llm, CASCADE_THRESHOLD, latency=latency),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark merchant categorization.")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--test", action="store_true", help="synthetic test set (once)")
    which.add_argument("--real", action="store_true", help="your hand-labeled set (once)")
    parser.add_argument("--force", action="store_true", help="allow a second --test/--real run")
    args = parser.parse_args()
    if args.test:
        _, _, test = merchant_split(load_synthetic())
        _run_once(TEST_REPORT, "synthetic test", test, latency=True, force=args.force)
    elif args.real:
        _run_once(REAL_REPORT, "your real transactions", load_private(), False, args.force)
    else:
        run_validation()


if __name__ == "__main__":
    main()
