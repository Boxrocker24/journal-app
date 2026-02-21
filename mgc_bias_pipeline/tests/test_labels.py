import pandas as pd

from src.labels import make_labels


def test_deadzone_labeling():
    bars = pd.DataFrame(
        {
            "session_id": ["NY_1", "NY_1", "NY_2", "NY_2", "NY_3", "NY_3"],
            "session_name": ["NY"] * 6,
            "ts_et": pd.date_range("2024-01-01", periods=6, freq="h", tz="America/New_York"),
            "open": [100, 100, 100, 100, 100, 100],
            "high": [101] * 6,
            "low": [99] * 6,
            "close": [101, 101, 99, 99, 100.02, 100.02],
        }
    )
    y = make_labels(bars, delta=0.005, drop_neutral=False)
    m = dict(zip(y["session_id"], y["y"]))
    assert m["NY_1"] == 1
    assert m["NY_2"] == 0
    assert pd.isna(m["NY_3"])
