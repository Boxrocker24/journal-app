from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.predict_core import compute_bias, load_model
from src.core.today_features import prepare_today_features
from src.utils.config import load_yaml
from src.utils.ui_helpers import append_signal_log_row


def generate_today_signal(
    *,
    sessions_cfg: str = "configs/sessions.yaml",
    features_cfg: str = "configs/features.yaml",
    model_cfg: str = "configs/model.yaml",
    model_path: str = "data/outputs/model.joblib",
    bars_with_session: str = "data/processed/bars_with_session.parquet",
    bars_raw: str = "data/raw/mgc_bars.parquet",
    signal_log_path: str = "data/outputs/today_signal_log.csv",
) -> dict[str, Any]:
    feat_result = prepare_today_features(
        sessions_cfg_path=sessions_cfg,
        features_cfg_path=features_cfg,
        bars_with_session_path=bars_with_session,
        raw_bars_path=bars_raw,
    )
    if feat_result["status"] != "ok":
        return {
            "prediction_ts_et": datetime.now().astimezone().isoformat(),
            "p_bull": None,
            "p_bear": None,
            "bias": "NEUTRAL",
            **{k: v for k, v in feat_result.items() if k != "features_df"},
        }

    cfg = load_yaml(model_cfg)
    bundle = load_model(model_path)
    latest = feat_result["features_df"]
    X = latest[[c for c in latest.columns if c not in {"y", "session_id", "start_ts_et"}]]
    p_bull = float(bundle["model"].predict_proba(X)[:, 1][0])
    th_long = cfg["decision"]["th_long"]
    th_short = cfg["decision"]["th_short"]
    bias = compute_bias(p_bull, th_long, th_short)
    p_bear = 1 - p_bull
    conf = max(p_bull, p_bear)
    model_version = str(bundle.get("trained_at", "unknown"))

    append_signal_log_row(
        signal_log_path,
        session_id=str(feat_result["session_id"]),
        p_bull=p_bull,
        p_bear=p_bear,
        conf=conf,
        bias=bias,
        th_long=float(th_long),
        th_short=float(th_short),
        model_version=model_version,
    )

    return {
        "session_id": feat_result["session_id"],
        "session_name": feat_result["session_name"],
        "start_ts_et": str(feat_result["start_ts_et"]),
        "cutoff_ts_et": str(feat_result["cutoff_ts_et"]),
        "prediction_ts_et": datetime.now().astimezone().isoformat(),
        "p_bull": p_bull,
        "p_bear": p_bear,
        "bias": bias,
        "conf": conf,
        "th_long": th_long,
        "th_short": th_short,
        "model_version": model_version,
        "status": "ok",
        "k_minutes": feat_result["k_minutes"],
        "required_bars": feat_result["required_bars"],
        "available_bars": feat_result["available_bars"],
        "source": feat_result["source"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sessions_cfg", default="configs/sessions.yaml")
    p.add_argument("--features_cfg", default="configs/features.yaml")
    p.add_argument("--model_cfg", default="configs/model.yaml")
    p.add_argument("--model_path", default="data/outputs/model.joblib")
    p.add_argument("--bars_with_session", default="data/processed/bars_with_session.parquet")
    p.add_argument("--bars", default="data/raw/mgc_bars.parquet")
    p.add_argument("--out", default="data/outputs/today_signal.json")
    p.add_argument("--signal_log", default="data/outputs/today_signal_log.csv")
    args = p.parse_args()

    payload = generate_today_signal(
        sessions_cfg=args.sessions_cfg,
        features_cfg=args.features_cfg,
        model_cfg=args.model_cfg,
        model_path=args.model_path,
        bars_with_session=args.bars_with_session,
        bars_raw=args.bars,
        signal_log_path=args.signal_log,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
