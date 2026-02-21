import csv

from src.utils.ui_helpers import append_signal_log_row


def test_append_signal_log_row_creates_header_and_appends(tmp_path):
    out_path = tmp_path / "today_signal_log.csv"

    append_signal_log_row(
        out_path,
        session_id="NY_2024-01-04",
        p_bull=0.61,
        p_bear=0.39,
        conf=0.61,
        bias="LONG",
        th_long=0.55,
        th_short=0.45,
        model_version="v1",
    )
    append_signal_log_row(
        out_path,
        session_id="NY_2024-01-05",
        p_bull=0.42,
        p_bear=0.58,
        conf=0.58,
        bias="SHORT",
        th_long=0.56,
        th_short=0.44,
        model_version="v2",
    )

    with out_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["session_id"] == "NY_2024-01-04"
    assert rows[0]["bias"] == "LONG"
    assert rows[0]["th_long"] == "0.55"
    assert rows[1]["session_id"] == "NY_2024-01-05"
    assert rows[1]["model_version"] == "v2"
