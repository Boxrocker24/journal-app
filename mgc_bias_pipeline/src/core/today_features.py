from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.features import build_session_summary, early_features, pre_session_features
from src.sessionize import assign_sessions
from src.utils.config import load_yaml


NON_FEATURE_COLUMNS = {"y", "session_id", "start_ts_et"}


def prepare_today_features(
    *,
    sessions_cfg_path: str | Path = "configs/sessions.yaml",
    features_cfg_path: str | Path = "configs/features.yaml",
    bars_with_session_path: str | Path = "data/processed/bars_with_session.parquet",
    raw_bars_path: str | Path = "data/raw/mgc_bars.parquet",
) -> dict[str, Any]:
    sessions_cfg = load_yaml(str(sessions_cfg_path))
    features_cfg = load_yaml(str(features_cfg_path))

    bars_path = Path(bars_with_session_path)
    if bars_path.exists():
        bars = pd.read_parquet(bars_path)
        source = "bars_with_session"
    else:
        raw_path = Path(raw_bars_path)
        if not raw_path.exists():
            return {"status": "missing_bars", "message": f"No bars file found at {bars_path} or {raw_path}."}
        raw_bars = pd.read_parquet(raw_path)
        bars, _ = assign_sessions(raw_bars, sessions_cfg)
        source = "raw_sessionized"

    summary = build_session_summary(bars)
    focus_sessions = sessions_cfg.get("focus_sessions", ["NY"])
    summary = summary[summary["session_name"].isin(focus_sessions)].sort_values("start_ts_et")
    if summary.empty:
        return {
            "status": "no_focus_session",
            "message": f"No sessions found for focus_sessions={focus_sessions}.",
            "focus_sessions": focus_sessions,
            "source": source,
        }

    latest = summary.iloc[-1]
    k_minutes = int(features_cfg["early_session"]["k_minutes"])
    bar_interval = int(features_cfg.get("bar_interval_minutes", 5))
    required_bars = max(1, math.ceil(k_minutes / bar_interval))
    cutoff_ts = latest["start_ts_et"] + timedelta(minutes=k_minutes)

    session_bars = bars[bars["session_id"] == latest["session_id"]].sort_values("ts_et")
    available_bars = int((session_bars["ts_et"] < cutoff_ts).sum())

    pre = pre_session_features(summary, features_cfg["pre_session"]["rolling_sessions"])
    early = early_features(bars, summary, k_minutes)
    base_cols = [
        "session_id",
        "session_name",
        "start_ts_et",
        "ret_prev",
        "range_prev",
        "roll_ret_mean",
        "roll_ret_std",
        "roll_range_mean",
        "dow",
        "hour",
    ]
    engineered = pre[base_cols].merge(early, on="session_id", how="left")
    latest_features = engineered[engineered["session_id"] == latest["session_id"]].copy()

    required_open = [c for c in base_cols if c not in {"session_id", "session_name", "start_ts_et"}]
    required_after = required_open + ["firstK_ret", "firstK_range", "time_above_open", "firstK_n"]

    if available_bars < required_bars:
        return {
            "status": "not_enough_bars",
            "message": "Not enough bars in current session to compute first-K early features.",
            "session_id": latest["session_id"],
            "session_name": latest["session_name"],
            "start_ts_et": latest["start_ts_et"],
            "cutoff_ts_et": cutoff_ts,
            "k_minutes": k_minutes,
            "bar_interval_minutes": bar_interval,
            "required_bars": required_bars,
            "available_bars": available_bars,
            "focus_sessions": focus_sessions,
            "source": source,
        }

    if latest_features.empty or latest_features[required_after].isna().any(axis=None):
        return {
            "status": "insufficient_history",
            "message": "Unable to build model-ready features for latest focus session (likely insufficient prior sessions).",
            "session_id": latest["session_id"],
            "session_name": latest["session_name"],
            "start_ts_et": latest["start_ts_et"],
            "cutoff_ts_et": cutoff_ts,
            "required_bars": required_bars,
            "available_bars": available_bars,
            "focus_sessions": focus_sessions,
            "source": source,
        }

    return {
        "status": "ok",
        "message": "Today features ready.",
        "session_id": latest["session_id"],
        "session_name": latest["session_name"],
        "start_ts_et": latest["start_ts_et"],
        "cutoff_ts_et": cutoff_ts,
        "k_minutes": k_minutes,
        "bar_interval_minutes": bar_interval,
        "required_bars": required_bars,
        "available_bars": available_bars,
        "focus_sessions": focus_sessions,
        "source": source,
        "features_df": latest_features.reset_index(drop=True),
        "feature_columns": [c for c in latest_features.columns if c not in NON_FEATURE_COLUMNS],
    }
