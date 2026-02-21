from __future__ import annotations

import argparse

import pandas as pd


def bucket_conf(x: float) -> str:
    if x < 0.55:
        return "<.55"
    if x < 0.60:
        return ".55-.60"
    if x < 0.65:
        return ".60-.65"
    if x < 0.70:
        return ".65-.70"
    if x < 0.80:
        return ".70-.80"
    return ">=.80"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", default="data/outputs/predictions.parquet")
    p.add_argument("--labels", default="data/processed/session_labels.parquet")
    p.add_argument("--out", default="data/outputs/eval_buckets.csv")
    args = p.parse_args()

    pred = pd.read_parquet(args.pred)
    labels = pd.read_parquet(args.labels)[["session_id", "y"]]
    df = pred.merge(labels, on="session_id", how="inner")
    traded = df[df["bias"].isin(["LONG", "SHORT"])].copy()
    traded["correct"] = ((traded["bias"] == "LONG") & (traded["y"] == 1)) | ((traded["bias"] == "SHORT") & (traded["y"] == 0))
    traded["conf"] = traded[["p_bull", "p_bear"]].max(axis=1)
    traded["bucket"] = traded["conf"].map(bucket_conf)

    buckets = traded.groupby("bucket", as_index=False).agg(n=("correct", "size"), precision=("correct", "mean"))
    buckets.to_csv(args.out, index=False)

    coverage = len(traded) / len(df) if len(df) else 0.0
    print({"coverage": coverage, "traded_precision": traded["correct"].mean() if len(traded) else 0.0})


if __name__ == "__main__":
    main()
