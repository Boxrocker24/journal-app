import pandas as pd

from src.features import build_features


def test_leakage_boundaries():
    rows = []
    base = pd.Timestamp("2024-01-01 08:20", tz="America/New_York")
    for i in range(3):
        sid = f"NY_2024-01-0{i+1}"
        start = base + pd.Timedelta(days=i)
        for j in range(6):
            rows.append(
                {
                    "session_id": sid,
                    "session_name": "NY",
                    "ts_et": start + pd.Timedelta(minutes=5 * j),
                    "open": 100 + i,
                    "high": 101 + i,
                    "low": 99 + i,
                    "close": 100 + i + (0.1 * j),
                }
            )
    bars = pd.DataFrame(rows)
    labels = pd.DataFrame(
        {
            "session_id": [f"NY_2024-01-0{i+1}" for i in range(3)],
            "y": [1, 0, 1],
        }
    )
    sessions_cfg = {"focus_sessions": ["NY"]}
    features_cfg = {"pre_session": {"rolling_sessions": 2}, "early_session": {"k_minutes": 15}}
    at_open, after_k = build_features(bars, labels, sessions_cfg, features_cfg)
    assert "firstK_ret" not in at_open.columns
    assert (after_k["firstK_n"] <= 3).all()
