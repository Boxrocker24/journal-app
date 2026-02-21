from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.utils.config import load_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", default="data/outputs/predictions.parquet")
    p.add_argument("--labels", default="data/processed/session_labels.parquet")
    p.add_argument("--features", default="data/processed/features_after_k.parquet")
    p.add_argument("--model_cfg", default="configs/model.yaml")
    p.add_argument("--out", default="data/outputs/monitor_report.csv")
    args = p.parse_args()

    cfg = load_yaml(args.model_cfg)
    pred = pd.read_parquet(args.pred)
    labels = pd.read_parquet(args.labels)[["session_id", "y"]]
    features = pd.read_parquet(args.features)
    df = pred.merge(labels, on="session_id", how="inner").sort_values("start_ts_et")

    traded = df[df["bias"].isin(["LONG", "SHORT"])].copy()
    traded["correct"] = ((traded["bias"] == "LONG") & (traded["y"] == 1)) | ((traded["bias"] == "SHORT") & (traded["y"] == 0))
    traded["rolling_precision_20"] = traded["correct"].rolling(20, min_periods=5).mean()
    df["coverage_roll_20"] = df["bias"].isin(["LONG", "SHORT"]).rolling(20, min_periods=5).mean()

    n = len(features)
    i = int(n * cfg["validation"]["train_frac"])
    num_cols = [c for c in features.columns if features[c].dtype.kind in "fi" and c != "y"]
    drift_rows = []
    for c in num_cols:
        train_s = features[c].iloc[:i]
        recent_s = features[c].iloc[max(i, n - 50) :]
        denom = train_s.std() if train_s.std() and not np.isnan(train_s.std()) else 1.0
        z_shift = (recent_s.mean() - train_s.mean()) / denom
        drift_rows.append({"feature": c, "z_shift": z_shift})
    drift_df = pd.DataFrame(drift_rows)

    report = traded[["session_id", "rolling_precision_20"]].merge(df[["session_id", "coverage_roll_20"]], on="session_id", how="left")
    report.to_csv(args.out, index=False)

    low_prec = traded["rolling_precision_20"].dropna().tail(1)
    if not low_prec.empty and low_prec.iloc[0] < cfg["monitor"]["min_precision_20"]:
        print("WARNING: rolling precision below threshold")
    drift_warn = drift_df[drift_df["z_shift"].abs() > cfg["monitor"]["max_feature_zshift"]]
    if not drift_warn.empty:
        print("WARNING: feature drift detected", drift_warn.to_dict(orient="records"))


if __name__ == "__main__":
    main()
