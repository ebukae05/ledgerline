"""Measure /score latency and throughput against a running API.

    python -m ledgerline.api.loadtest --url http://localhost:8000

Replays real test-split transactions at a few concurrency levels and reports
p50/p95/p99 latency and requests per second, with flagged and unflagged
requests also measured separately (flagged ones compute reasons, which is
slower). Writes reports/loadtest.json.
"""

import argparse
import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import httpx
import numpy as np

from ledgerline.data.load import PROJECT_ROOT, load_train
from ledgerline.data.split import time_split

REPORT = PROJECT_ROOT / "reports" / "loadtest.json"
SEED = 0


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def payloads(n: int, raw_columns: set[str]) -> list[dict]:
    _, _, test = time_split(load_train())
    rows = test.sample(n, seed=SEED, with_replacement=n > test.height)
    return [
        {
            k: v
            for k, v in row.items()
            if k in raw_columns and v is not None and not (isinstance(v, float) and np.isnan(v))
        }
        for row in rows.iter_rows(named=True)
    ]


def run(url: str, bodies: list[dict], concurrency: int) -> dict:
    client = httpx.Client(
        base_url=url, timeout=60, limits=httpx.Limits(max_connections=concurrency)
    )

    def call(body):
        start = time.perf_counter()
        r = client.post("/score", json=body)
        elapsed = (time.perf_counter() - start) * 1000
        flagged = r.status_code == 200 and r.json()["flagged"]
        return elapsed, r.status_code, flagged

    started = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        results = list(pool.map(call, bodies))
    wall = time.perf_counter() - started
    client.close()

    ok = [r for r in results if r[1] == 200]
    all_ms = [r[0] for r in ok]
    flagged_ms = [r[0] for r in ok if r[2]]
    plain_ms = [r[0] for r in ok if not r[2]]
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "errors": len(results) - len(ok),
        "requests_per_second": len(results) / wall,
        "p50_ms": _pct(all_ms, 50),
        "p95_ms": _pct(all_ms, 95),
        "p99_ms": _pct(all_ms, 99),
        "unflagged_p50_ms": _pct(plain_ms, 50),
        "unflagged_p95_ms": _pct(plain_ms, 95),
        "flagged_count": len(flagged_ms),
        "flagged_p50_ms": _pct(flagged_ms, 50),
        "flagged_p95_ms": _pct(flagged_ms, 95),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Load-test /score.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=1_000)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--note", default="", help="describe the setup, e.g. '2 workers'")
    args = parser.parse_args()

    from ledgerline.api.scoring import Scorer
    from ledgerline.models.registry import load_model

    model, meta = load_model()
    columns = set(Scorer(model, meta).schema) | {"TransactionID", "TransactionDT"}
    bodies = payloads(args.requests + args.warmup, columns)
    health = httpx.get(f"{args.url}/health", timeout=10).json()

    run(args.url, bodies[: args.warmup], 1)
    levels = []
    for c in args.concurrency:
        result = run(args.url, bodies[args.warmup :], c)
        levels.append(result)
        print(
            f"concurrency {c:>2}: {result['requests_per_second']:6.1f} req/s  "
            f"p50 {result['p50_ms']:6.1f} ms  p95 {result['p95_ms']:6.1f} ms  "
            f"p99 {result['p99_ms']:6.1f} ms  errors {result['errors']}  "
            f"(flagged: {result['flagged_count']}, p50 {result['flagged_p50_ms']:.0f} ms)",
            flush=True,
        )
    report = {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "url": args.url,
        "model_version": health.get("model_version"),
        "note": args.note,
        "machine": {"cpus": os.cpu_count(), "platform": platform.platform()},
        "levels": levels,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {REPORT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
