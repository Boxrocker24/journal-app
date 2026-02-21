from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


SIGNAL_LOG_COLUMNS = [
    "timestamp_et",
    "session_id",
    "p_bull",
    "p_bear",
    "conf",
    "bias",
    "th_long",
    "th_short",
    "model_version",
]


def append_signal_log_row(
    out_path: str | Path,
    *,
    session_id: str,
    p_bull: float,
    p_bear: float,
    conf: float,
    bias: str,
    th_long: float,
    th_short: float,
    model_version: str,
) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    timestamp_et = datetime.now(ZoneInfo("America/New_York")).isoformat()

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SIGNAL_LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_et": timestamp_et,
                "session_id": session_id,
                "p_bull": p_bull,
                "p_bear": p_bear,
                "conf": conf,
                "bias": bias,
                "th_long": th_long,
                "th_short": th_short,
                "model_version": model_version,
            }
        )
