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
