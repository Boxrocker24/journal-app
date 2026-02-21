import pandas as pd

from src.sessionize import assign_sessions


def test_cross_midnight_assignment():
    ts = pd.to_datetime([
        "2024-01-01 23:00:00+00:00",
        "2024-01-02 00:30:00+00:00",
    ])
    bars = pd.DataFrame(
        {
            "ts_et": ts.tz_convert("America/New_York"),
            "open": [1, 1],
            "high": [1, 1],
            "low": [1, 1],
            "close": [1, 1],
        }
    )
    cfg = {
        "timezone": "America/New_York",
        "sessions": [{"name": "ASIA", "start": "18:00", "end": "02:00"}],
    }
    out, _ = assign_sessions(bars, cfg)
    assert out["session_name"].eq("ASIA").all()
    assert out["session_id"].nunique() == 1
