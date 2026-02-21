from __future__ import annotations

import argparse

import pandas as pd

from src.core.predict_core import compute_bias, load_model, predict_from_features_df
from src.utils.config import load_yaml


def apply_bias(p_bull: float, th_long: float, th_short: float) -> str:
    return compute_bias(p_bull, th_long, th_short)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/features_after_k.parquet")
    p.add_argument("--model_path", default="data/outputs/model.joblib")
    p.add_argument("--model_cfg", default="configs/model.yaml")
    p.add_argument("--out", default="data/outputs/predictions.parquet")
    args = p.parse_args()

    df = pd.read_parquet(args.features)
    bundle = load_model(args.model_path)
    model = bundle["model"]
    cfg = load_yaml(args.model_cfg)

    out = predict_from_features_df(df, model, bundle["trained_at"], cfg["decision"])
    out.to_parquet(args.out, index=False)


if __name__ == "__main__":
    main()
