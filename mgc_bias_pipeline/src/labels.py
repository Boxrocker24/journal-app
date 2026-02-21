from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.utils.config import load_yaml


def make_labels(bars: pd.DataFrame, delta: float, drop_neutral: bool) -> pd.DataFrame:
    g = bars.sort_values("ts_et").groupby(["session_id", "session_name"], as_index=False)
    out = g.agg(
        start_ts_et=("ts_et", "min"),
        open_s=("open", "first"),
        close_s=("close", "last"),
        high_s=("high", "max"),
        low_s=("low", "min"),
        n_bars=("close", "size"),
    )
    out["ret_s"] = (out["close_s"] - out["open_s"]) / out["open_s"]
    out["y"] = np.where(out["ret_s"] >= delta, 1, np.where(out["ret_s"] <= -delta, 0, np.nan))
    if drop_neutral:
        out = out.dropna(subset=["y"]).copy()
    out["y"] = out["y"].astype(int)
    return out.sort_values("start_ts_et")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/bars_with_session.parquet")
    p.add_argument("--model_cfg", default="configs/model.yaml")
    p.add_argument("--out", default="data/processed/session_labels.parquet")
    args = p.parse_args()
    bars = pd.read_parquet(args.input)
    cfg = load_yaml(args.model_cfg)
    labels = make_labels(bars, cfg["label"]["delta"], cfg["label"]["drop_neutral"])
    labels.to_parquet(args.out, index=False)


if __name__ == "__main__":
    main()
