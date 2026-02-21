from __future__ import annotations

import argparse
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.config import load_yaml


def time_split(df: pd.DataFrame, train_frac: float, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("start_ts_et").reset_index(drop=True)
    n = len(df)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def get_xy(df: pd.DataFrame, y_col: str = "y"):
    features = [c for c in df.columns if c not in {"y", "session_id", "start_ts_et"}]
    X = df[features]
    y = df[y_col]
    cat_cols = [c for c in ["session_name"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return X, y, cat_cols, num_cols


def metrics(y_true, p):
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "logloss": log_loss(y_true, p, labels=[0, 1]),
        "brier": brier_score_loss(y_true, p),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/features_after_k.parquet")
    p.add_argument("--model_cfg", default="configs/model.yaml")
    p.add_argument("--out_model", default="data/outputs/model.joblib")
    p.add_argument("--out_report", default="data/outputs/train_report.csv")
    args = p.parse_args()

    cfg = load_yaml(args.model_cfg)
    df = pd.read_parquet(args.features)
    tr, va, te = time_split(df, cfg["validation"]["train_frac"], cfg["validation"]["val_frac"])
    Xtr, ytr, cat_cols, num_cols = get_xy(tr)
    Xva, yva, _, _ = get_xy(va)
    Xte, yte, _, _ = get_xy(te)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    lr = Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=1000))])
    lr.fit(Xtr, ytr)

    gb_base = Pipeline(
        [("prep", preprocess), ("clf", HistGradientBoostingClassifier(random_state=42, max_depth=4))]
    )
    gb_base.fit(Xtr, ytr)
    gb_cal = CalibratedClassifierCV(gb_base, method="isotonic", cv="prefit")
    gb_cal.fit(Xva, yva)

    use_gb = cfg["model"].get("gradient_boosting", True)
    model = gb_cal if use_gb else lr
    mtype = "calibrated_hgb" if use_gb else "logistic"

    report_rows = []
    for name, X, y in [("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)]:
        p1 = model.predict_proba(X)[:, 1]
        mm = metrics(y, p1)
        mm["split"] = name
        report_rows.append(mm)
    pd.DataFrame(report_rows).to_csv(args.out_report, index=False)

    bundle = {
        "model": model,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "model_type": mtype,
        "trained_at": datetime.utcnow().isoformat(),
        "feature_set": args.features,
    }
    joblib.dump(bundle, args.out_model)


if __name__ == "__main__":
    main()
