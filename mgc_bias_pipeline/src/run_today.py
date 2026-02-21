from __future__ import annotations

import argparse
import json
from datetime import datetime

import joblib
import pandas as pd

from src.features import build_features
from src.labels import make_labels
from src.sessionize import assign_sessions
from src.utils.config import load_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bars", default="data/raw/mgc_bars.parquet")
    p.add_argument("--sessions_cfg", default="configs/sessions.yaml")
    p.add_argument("--features_cfg", default="configs/features.yaml")
    p.add_argument("--model_cfg", default="configs/model.yaml")
    p.add_argument("--model_path", default="data/outputs/model.joblib")
    p.add_argument("--out", default="data/outputs/today_signal.json")
    args = p.parse_args()

    bars = pd.read_parquet(args.bars)
    sessions_cfg = load_yaml(args.sessions_cfg)
    features_cfg = load_yaml(args.features_cfg)
    model_cfg = load_yaml(args.model_cfg)

    bars_s, _ = assign_sessions(bars, sessions_cfg)
    labels = make_labels(bars_s, model_cfg["label"]["delta"], model_cfg["label"]["drop_neutral"])
    _, after_k = build_features(bars_s, labels, sessions_cfg, features_cfg)
    latest = after_k.sort_values("start_ts_et").tail(1)

    bundle = joblib.load(args.model_path)
    X = latest[[c for c in latest.columns if c not in {"y", "session_id", "start_ts_et"}]]
    p_bull = float(bundle["model"].predict_proba(X)[:, 1][0])
    th_long = model_cfg["decision"]["th_long"]
    th_short = model_cfg["decision"]["th_short"]
    bias = "LONG" if p_bull >= th_long else ("SHORT" if (1 - p_bull) >= th_short else "NEUTRAL")

    payload = {
        "session_id": latest["session_id"].iloc[0],
        "prediction_ts_et": datetime.now().astimezone().isoformat(),
        "p_bull": p_bull,
        "p_bear": 1 - p_bull,
        "bias": bias,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
