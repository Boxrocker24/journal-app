import pytest
import csv

import numpy as np
import pandas as pd

from src import run_today


class _DummyModel:
    def predict_proba(self, X):
        return np.array([[0.2, 0.8] for _ in range(len(X))])


def test_generate_today_signal_appends_signal_log(monkeypatch, tmp_path):
    monkeypatch.setattr(
        run_today,
        "prepare_today_features",
        lambda **kwargs: {
            "status": "ok",
            "features_df": pd.DataFrame([{"session_id": "NY_2024-01-04", "start_ts_et": "2024-01-04T08:20:00-05:00", "f1": 1.0}]),
            "session_id": "NY_2024-01-04",
            "session_name": "NY",
            "start_ts_et": "2024-01-04T08:20:00-05:00",
            "cutoff_ts_et": "2024-01-04T08:50:00-05:00",
            "k_minutes": 30,
            "required_bars": 6,
            "available_bars": 6,
            "source": "bars_with_session",
        },
    )
    monkeypatch.setattr(run_today, "load_yaml", lambda _: {"decision": {"th_long": 0.6, "th_short": 0.4}})
    monkeypatch.setattr(run_today, "load_model", lambda _: {"model": _DummyModel(), "trained_at": "2026-01-01"})

    out_log = tmp_path / "today_signal_log.csv"
    payload = run_today.generate_today_signal(signal_log_path=str(out_log))

    with out_log.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert payload["p_bull"] == 0.8
    assert payload["p_bear"] == pytest.approx(0.2)
    assert payload["conf"] == pytest.approx(0.8)
    assert payload["bias"] == "LONG"
    assert payload["th_long"] == 0.6
    assert payload["th_short"] == 0.4
    assert payload["model_version"] == "2026-01-01"
    assert len(rows) == 1
    assert rows[0]["session_id"] == "NY_2024-01-04"
    assert rows[0]["th_long"] == "0.6"
    assert rows[0]["model_version"] == "2026-01-01"
