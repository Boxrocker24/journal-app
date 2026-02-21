from __future__ import annotations

import argparse
from datetime import timedelta

import pandas as pd

from src.utils.config import load_yaml
from src.utils.logging import get_logger
from src.utils.time import build_window

logger = get_logger(__name__)


def assign_sessions(bars: pd.DataFrame, sessions_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    timezone = sessions_cfg["timezone"]
    sessions = sessions_cfg["sessions"]
    bars = bars.sort_values("ts_et").copy()
    bars["session_id"] = pd.NA
    bars["session_name"] = pd.NA
    bars["session_start_et"] = pd.NaT

    min_day = bars["ts_et"].min().floor("D") - timedelta(days=1)
    max_day = bars["ts_et"].max().floor("D") + timedelta(days=1)
    all_days = pd.date_range(min_day, max_day, freq="D", tz=bars["ts_et"].dt.tz)

    session_rows: list[dict] = []
    for day in all_days:
        for s in sessions:
            start, end = build_window(day, s["start"], s["end"], timezone)
            session_id = f"{s['name']}_{start.date()}"
            mask = (bars["ts_et"] >= start) & (bars["ts_et"] < end) & bars["session_id"].isna()
            if mask.any():
                bars.loc[mask, "session_id"] = session_id
                bars.loc[mask, "session_name"] = s["name"]
                bars.loc[mask, "session_start_et"] = start
                session_rows.append(
                    {
                        "session_id": session_id,
                        "session_name": s["name"],
                        "start_ts_et": start,
                        "end_ts_et": end,
                    }
                )

    bars = bars.dropna(subset=["session_id"]).copy()
    bars["bar_idx_in_session"] = bars.groupby("session_id").cumcount()
    session_index = pd.DataFrame(session_rows).drop_duplicates(subset=["session_id"]).sort_values("start_ts_et")
    return bars, session_index


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/mgc_bars.parquet")
    p.add_argument("--sessions_cfg", default="configs/sessions.yaml")
    p.add_argument("--out_bars", default="data/processed/bars_with_session.parquet")
    p.add_argument("--out_index", default="data/processed/session_index.parquet")
    args = p.parse_args()

    bars = pd.read_parquet(args.input)
    cfg = load_yaml(args.sessions_cfg)
    out_bars, out_idx = assign_sessions(bars, cfg)
    out_bars.to_parquet(args.out_bars, index=False)
    out_idx.to_parquet(args.out_index, index=False)
    logger.info("Wrote %s and %s", args.out_bars, args.out_index)


if __name__ == "__main__":
    main()
