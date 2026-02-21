from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


NON_FEATURE_COLUMNS = {"y", "session_id", "start_ts_et"}


def load_model(model_path: str | Path) -> Any:
    return joblib.load(model_path)


def compute_bias(p_bull: float, th_long: float, th_short: float) -> str:
    if p_bull >= th_long:
        return "LONG"
    if p_bull <= th_short:
        return "SHORT"
    return "NEUTRAL"


def predict_from_features_df(
    features_df: pd.DataFrame,
    model: Any,
    model_version: str,
    thresholds: Mapping[str, float],
) -> pd.DataFrame:
    X = features_df[[c for c in features_df.columns if c not in NON_FEATURE_COLUMNS]]
    p_bull = model.predict_proba(X)[:, 1]

    out = features_df[["session_id", "session_name", "start_ts_et"]].copy()
    out["p_bull"] = p_bull
    out["p_bear"] = 1 - p_bull
    out["conf"] = out[["p_bull", "p_bear"]].max(axis=1)
    out["bias"] = [compute_bias(v, thresholds["th_long"], thresholds["th_short"]) for v in out["p_bull"]]
    out["model_version"] = model_version
    return out
