from __future__ import annotations

import argparse

import joblib
import pandas as pd

from src.utils.config import load_yaml


def apply_bias(p_bull: float, th_long: float, th_short: float) -> str:
    if p_bull >= th_long:
        return "LONG"
    if (1 - p_bull) >= th_short:
        return "SHORT"
    return "NEUTRAL"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/features_after_k.parquet")
    p.add_argument("--model_path", default="data/outputs/model.joblib")
    p.add_argument("--model_cfg", default="configs/model.yaml")
    p.add_argument("--out", default="data/outputs/predictions.parquet")
    args = p.parse_args()

    df = pd.read_parquet(args.features)
    bundle = joblib.load(args.model_path)
    model = bundle["model"]
    cfg = load_yaml(args.model_cfg)

    X = df[[c for c in df.columns if c not in {"y", "session_id", "start_ts_et"}]]
    p_bull = model.predict_proba(X)[:, 1]
    out = df[["session_id", "session_name", "start_ts_et"]].copy()
    out["p_bull"] = p_bull
    out["p_bear"] = 1 - p_bull
    out["bias"] = [
        apply_bias(v, cfg["decision"]["th_long"], cfg["decision"]["th_short"]) for v in out["p_bull"]
    ]
    out["model_version"] = bundle["trained_at"]
    out.to_parquet(args.out, index=False)


if __name__ == "__main__":
    main()
