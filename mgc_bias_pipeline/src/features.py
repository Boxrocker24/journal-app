from __future__ import annotations

import argparse
from datetime import timedelta

import numpy as np
import pandas as pd

from src.utils.config import load_yaml


def build_session_summary(bars: pd.DataFrame) -> pd.DataFrame:
    g = bars.sort_values("ts_et").groupby(["session_id", "session_name"], as_index=False)
    s = g.agg(
        start_ts_et=("ts_et", "min"),
        end_ts_et=("ts_et", "max"),
        open_s=("open", "first"),
        close_s=("close", "last"),
        high_s=("high", "max"),
        low_s=("low", "min"),
    ).sort_values("start_ts_et")
    s["ret_s"] = (s["close_s"] - s["open_s"]) / s["open_s"]
    s["range_s"] = (s["high_s"] - s["low_s"]) / s["open_s"]
    return s


def pre_session_features(summary: pd.DataFrame, rolling_n: int) -> pd.DataFrame:
    df = summary.copy().sort_values("start_ts_et")
    df["ret_prev"] = df["ret_s"].shift(1)
    df["range_prev"] = df["range_s"].shift(1)
    df["roll_ret_mean"] = df["ret_s"].shift(1).rolling(rolling_n, min_periods=3).mean()
    df["roll_ret_std"] = df["ret_s"].shift(1).rolling(rolling_n, min_periods=3).std()
    df["roll_range_mean"] = df["range_s"].shift(1).rolling(rolling_n, min_periods=3).mean()
    df["dow"] = df["start_ts_et"].dt.dayofweek
    df["hour"] = df["start_ts_et"].dt.hour
    return df


def early_features(bars: pd.DataFrame, summary: pd.DataFrame, k_minutes: int) -> pd.DataFrame:
    rows = []
    for row in summary.itertuples(index=False):
        cutoff = row.start_ts_et + timedelta(minutes=k_minutes)
        b = bars[(bars["session_id"] == row.session_id) & (bars["ts_et"] < cutoff)].sort_values("ts_et")
        if b.empty:
            rows.append({"session_id": row.session_id, "firstK_n": 0})
            continue
        open_s = row.open_s
        rows.append(
            {
                "session_id": row.session_id,
                "firstK_ret": (b["close"].iloc[-1] - open_s) / open_s,
                "firstK_range": (b["high"].max() - b["low"].min()) / open_s,
                "time_above_open": (b["close"] > open_s).mean(),
                "firstK_n": len(b),
            }
        )
    return pd.DataFrame(rows)


def build_features(bars: pd.DataFrame, labels: pd.DataFrame, sessions_cfg: dict, features_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = build_session_summary(bars)
    focus = set(sessions_cfg.get("focus_sessions", ["NY"]))
    summary = summary[summary["session_name"].isin(focus)].copy()
    pre = pre_session_features(summary, features_cfg["pre_session"]["rolling_sessions"])
    early = early_features(bars, summary, features_cfg["early_session"]["k_minutes"])

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
    at_open = pre[base_cols].merge(labels[["session_id", "y"]], on="session_id", how="left")
    after_k = pre[base_cols].merge(early, on="session_id", how="left").merge(labels[["session_id", "y"]], on="session_id", how="left")

    required_open = [c for c in base_cols if c not in {"session_id", "session_name", "start_ts_et"}]
    required_after = required_open + ["firstK_ret", "firstK_range", "time_above_open", "firstK_n"]
    at_open = at_open.dropna(subset=required_open + ["y"]).reset_index(drop=True)
    after_k = after_k.dropna(subset=required_after + ["y"]).reset_index(drop=True)
    at_open["y"] = at_open["y"].astype(int)
    after_k["y"] = after_k["y"].astype(int)
    return at_open, after_k


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bars", default="data/processed/bars_with_session.parquet")
    p.add_argument("--labels", default="data/processed/session_labels.parquet")
    p.add_argument("--sessions_cfg", default="configs/sessions.yaml")
    p.add_argument("--features_cfg", default="configs/features.yaml")
    p.add_argument("--out_open", default="data/processed/features_at_open.parquet")
    p.add_argument("--out_k", default="data/processed/features_after_k.parquet")
    args = p.parse_args()

    bars = pd.read_parquet(args.bars)
    labels = pd.read_parquet(args.labels)
    sessions_cfg = load_yaml(args.sessions_cfg)
    features_cfg = load_yaml(args.features_cfg)
    at_open, after_k = build_features(bars, labels, sessions_cfg, features_cfg)
    at_open.to_parquet(args.out_open, index=False)
    after_k.to_parquet(args.out_k, index=False)


if __name__ == "__main__":
    main()
