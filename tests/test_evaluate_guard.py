import json

import pytest

from ledgerline import evaluate


def test_refuses_to_evaluate_test_set_twice(tmp_path, monkeypatch):
    report = tmp_path / "test_results.json"
    report.write_text(json.dumps({"model_version": "lgbm-abc", "evaluated_at": "2026-01-01"}))
    monkeypatch.setattr(evaluate, "REPORT_PATH", report)
    monkeypatch.setattr(evaluate, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["evaluate"])

    with pytest.raises(SystemExit, match="already evaluated for lgbm-abc"):
        evaluate.main()


@pytest.mark.parametrize("flag", ["--test", "--real"])
def test_merchant_benchmark_refuses_a_second_run(tmp_path, monkeypatch, flag):
    from ledgerline.merchants import benchmark

    for name in ("TEST_REPORT", "REAL_REPORT"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}")
        monkeypatch.setattr(benchmark, name, path)
    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "load_private", lambda: None)
    monkeypatch.setattr(benchmark, "merchant_split", lambda df: (None, None, None))
    monkeypatch.setattr(benchmark, "load_synthetic", lambda: None)
    monkeypatch.setattr("sys.argv", ["benchmark", flag])

    with pytest.raises(SystemExit, match="already evaluated"):
        benchmark.main()
