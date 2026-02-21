from __future__ import annotations

import pandas as pd


REQUIRED_OHLC = ["open", "high", "low", "close"]


def normalize_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    out = df.copy()
    for req in REQUIRED_OHLC:
        if req not in cols:
            raise ValueError(f"Missing required column: {req}")
        out.rename(columns={cols[req]: req}, inplace=True)
    ts_candidates = ["ts", "timestamp", "datetime"]
    match = next((c for c in ts_candidates if c in cols), None)
    if match is None:
        raise ValueError("Missing timestamp column; expected one of ts/timestamp/datetime")
    out.rename(columns={cols[match]: "ts_raw"}, inplace=True)
    return out


def clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.dropna(subset=["ts_et", "open", "high", "low", "close"]).copy()
    out = out[(out["high"] >= out[["open", "close"]].max(axis=1)) & (out["low"] <= out[["open", "close"]].min(axis=1))]
    out = out[out["high"] >= out["low"]]
    return out
