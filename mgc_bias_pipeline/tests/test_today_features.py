import pandas as pd

from src.core.today_features import prepare_today_features


def test_prepare_today_features_reports_not_enough_bars(tmp_path):
    start = pd.Timestamp("2024-01-04 08:20", tz="America/New_York")
    bars = pd.DataFrame(
        [
            {
                "session_id": "NY_2024-01-04",
                "session_name": "NY",
                "ts_et": start + pd.Timedelta(minutes=5 * i),
                "open": 100.0,
                "high": 100.2 + i,
                "low": 99.9,
                "close": 100.1 + i * 0.1,
            }
            for i in range(3)
        ]
    )
    bars_path = tmp_path / "bars_with_session.parquet"
    bars.to_parquet(bars_path, index=False)

    sessions_cfg = tmp_path / "sessions.yaml"
    sessions_cfg.write_text(
        """
timezone: America/New_York
sessions:
  - name: NY
    start: \"08:20\"
    end: \"17:00\"
focus_sessions: [\"NY\"]
""".strip(),
        encoding="utf-8",
    )
    features_cfg = tmp_path / "features.yaml"
    features_cfg.write_text(
        """
pre_session:
  rolling_sessions: 3
early_session:
  k_minutes: 30
bar_interval_minutes: 5
""".strip(),
        encoding="utf-8",
    )

    out = prepare_today_features(
        sessions_cfg_path=sessions_cfg,
        features_cfg_path=features_cfg,
        bars_with_session_path=bars_path,
        raw_bars_path=tmp_path / "missing.parquet",
    )

    assert out["status"] == "not_enough_bars"
    assert out["available_bars"] == 3
    assert out["required_bars"] == 6
