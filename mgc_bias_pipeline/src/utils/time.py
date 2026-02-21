from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from zoneinfo import ZoneInfo


def to_et(series: pd.Series, tz_name: str = "America/New_York") -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    tz = ZoneInfo(tz_name)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert(tz)
    else:
        ts = ts.dt.tz_convert(tz)
    return ts


def parse_hhmm(hhmm: str) -> tuple[int, int]:
    hh, mm = hhmm.split(":")
    return int(hh), int(mm)


def build_window(start_date: pd.Timestamp, start_hhmm: str, end_hhmm: str, tz_name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    tz = ZoneInfo(tz_name)
    sh, sm = parse_hhmm(start_hhmm)
    eh, em = parse_hhmm(end_hhmm)
    start_dt = datetime(start_date.year, start_date.month, start_date.day, sh, sm, tzinfo=tz)
    end_dt = datetime(start_date.year, start_date.month, start_date.day, eh, em, tzinfo=tz)
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)
    return pd.Timestamp(start_dt), pd.Timestamp(end_dt)
