from __future__ import annotations

import argparse
import json
import subprocess
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

import pandas as pd

from src.predict import apply_bias
from src.run_today import generate_today_signal
from src.utils.config import load_yaml


class AppUI(tk.Tk):
    def __init__(self, root_dir: Path):
        super().__init__()
        self.root_dir = root_dir
        self.title("MGC Bias Control Center")
        self.geometry("1280x820")

        self.model_cfg_path = self.root_dir / "configs/model.yaml"
        self.today_path = self.root_dir / "data/outputs/today_signal.json"
        self.pred_path = self.root_dir / "data/outputs/predictions.parquet"
        self.labels_path = self.root_dir / "data/processed/session_labels.parquet"
        self.features_path = self.root_dir / "data/processed/features_after_k.parquet"

        cfg = load_yaml(str(self.model_cfg_path)) if self.model_cfg_path.exists() else {"decision": {"th_long": 0.55, "th_short": 0.55}}

        self.today_signal: dict[str, Any] = {}
        self.recent_df = pd.DataFrame()
        self.stats_df = pd.DataFrame()

        self.th_long = tk.DoubleVar(value=float(cfg["decision"]["th_long"]))
        self.th_short = tk.DoubleVar(value=float(cfg["decision"]["th_short"]))
        self.bias_var = tk.StringVar(value="--")
        self.freshness_var = tk.StringVar(value="unknown")
        self.meta_var = tk.StringVar(value="No session loaded")
        self.pbull_var = tk.StringVar(value="p_bull: --")
        self.pbear_var = tk.StringVar(value="p_bear: --")
        self.conf_var = tk.StringVar(value="confidence: --")
        self.status_var = tk.StringVar(value="Ready")
        self.filter_bias = tk.StringVar(value="ALL")
        self.last_n = tk.IntVar(value=20)
        self.include_train = tk.BooleanVar(value=False)

        self._build_ui()
        self.refresh_data()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        signal = ttk.LabelFrame(self, text="1) Today Signal")
        signal.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        signal.columnconfigure(0, weight=1)

        ttk.Label(signal, textvariable=self.meta_var).grid(row=0, column=0, sticky="w", padx=8, pady=2)
        ttk.Label(signal, textvariable=self.pbull_var).grid(row=1, column=0, sticky="w", padx=8)
        ttk.Label(signal, textvariable=self.pbear_var).grid(row=2, column=0, sticky="w", padx=8)
        ttk.Label(signal, textvariable=self.conf_var).grid(row=3, column=0, sticky="w", padx=8)

        self.bias_label = ttk.Label(signal, textvariable=self.bias_var, font=("TkDefaultFont", 24, "bold"))
        self.bias_label.grid(row=0, column=1, rowspan=3, sticky="e", padx=12)
        ttk.Label(signal, textvariable=self.freshness_var).grid(row=3, column=1, sticky="e", padx=12)

        controls = ttk.LabelFrame(self, text="2) Threshold Controls")
        controls.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        controls.columnconfigure(7, weight=1)

        ttk.Label(controls, text="th_long").grid(row=0, column=0, padx=4)
        long_scale = ttk.Scale(controls, from_=0.4, to=0.8, variable=self.th_long, command=lambda _: self.recompute_bias())
        long_scale.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Entry(controls, textvariable=self.th_long, width=7).grid(row=0, column=2, padx=4)

        ttk.Label(controls, text="th_short").grid(row=0, column=3, padx=4)
        short_scale = ttk.Scale(controls, from_=0.4, to=0.8, variable=self.th_short, command=lambda _: self.recompute_bias())
        short_scale.grid(row=0, column=4, sticky="ew", padx=4)
        ttk.Entry(controls, textvariable=self.th_short, width=7).grid(row=0, column=5, padx=4)

        ttk.Button(controls, text="Balanced", command=lambda: self.set_preset(0.55, 0.55)).grid(row=0, column=6, padx=2)
        ttk.Button(controls, text="Conservative", command=lambda: self.set_preset(0.62, 0.62)).grid(row=0, column=7, padx=2, sticky="w")
        ttk.Button(controls, text="Aggressive", command=lambda: self.set_preset(0.51, 0.51)).grid(row=0, column=8, padx=2)

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)

        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=2)

        table_frame = ttk.LabelFrame(left, text="3) Recent Signals")
        table_frame.pack(fill=tk.BOTH, expand=True)

        filters = ttk.Frame(table_frame)
        filters.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(filters, text="Last N").pack(side=tk.LEFT)
        ttk.Spinbox(filters, from_=5, to=200, textvariable=self.last_n, width=6, command=self.apply_filters).pack(side=tk.LEFT, padx=4)
        ttk.Label(filters, text="Bias filter").pack(side=tk.LEFT)
        ttk.Combobox(filters, textvariable=self.filter_bias, values=["ALL", "LONG", "SHORT", "NEUTRAL"], width=10).pack(side=tk.LEFT, padx=4)
        ttk.Button(filters, text="Apply", command=self.apply_filters).pack(side=tk.LEFT, padx=4)
        ttk.Button(filters, text="Replay Selected", command=self.replay_selected).pack(side=tk.LEFT, padx=4)

        cols = ("session_id", "start_ts_et", "p_bull", "p_bear", "bias")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=18)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=130 if c != "session_id" else 220)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        stats = ttk.LabelFrame(right, text="4) Performance Quick Stats")
        stats.pack(fill=tk.BOTH, expand=True)
        self.stats_text = tk.Text(stats, height=12, width=44)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        actions = ttk.LabelFrame(right, text="5) Actions")
        actions.pack(fill=tk.X, pady=6)
        ttk.Button(actions, text="Refresh Data", command=self.refresh_data).pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(actions, text="Open Replay", command=self.open_replay).pack(fill=tk.X, padx=6, pady=4)
        ttk.Checkbutton(actions, text="Include training when pipeline runs", variable=self.include_train).pack(fill=tk.X, padx=6)
        ttk.Button(actions, text="Run Full Pipeline", command=self.run_pipeline).pack(fill=tk.X, padx=6, pady=4)

        ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w").grid(row=3, column=0, sticky="ew")

    def set_status(self, msg: str) -> None:
        self.status_var.set(msg)

    def set_preset(self, th_long: float, th_short: float) -> None:
        self.th_long.set(th_long)
        self.th_short.set(th_short)
        self.recompute_bias()

    def run_in_bg(self, label: str, task, on_success=None) -> None:
        self.set_status(f"{label}...")

        def worker() -> None:
            try:
                result = task()
                self.after(0, lambda: self._on_task_success(label, result, on_success))
            except Exception as exc:  # broad to keep UI responsive
                self.after(0, lambda: self._on_task_error(label, exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_task_success(self, label: str, result: Any, callback) -> None:
        self.set_status(f"{label} complete")
        if callback:
            callback(result)

    def _on_task_error(self, label: str, exc: Exception) -> None:
        self.set_status(f"{label} failed")
        messagebox.showerror("Error", f"{label} failed: {exc}")

    def freshness(self, ts: str | None) -> str:
        if not ts:
            return "unknown"
        try:
            dt = datetime.fromisoformat(ts)
            age = datetime.now().astimezone() - dt
            if age.total_seconds() < 3600:
                return "fresh (<1h)"
            if age.total_seconds() < 86400:
                return "stale (>1h)"
            return "old (>24h)"
        except Exception:
            return "unknown"

    def refresh_data(self) -> None:
        def task() -> dict[str, Any]:
            today = {}
            if self.today_path.exists():
                today = json.loads(self.today_path.read_text(encoding="utf-8"))
            try:
                live_today = generate_today_signal(
                    sessions_cfg="configs/sessions.yaml",
                    features_cfg="configs/features.yaml",
                    model_cfg=str(self.model_cfg_path),
                    model_path="data/outputs/model.joblib",
                    bars_with_session="data/processed/bars_with_session.parquet",
                    bars_raw="data/raw/mgc_bars.parquet",
                )
                today = live_today
                self.today_path.parent.mkdir(parents=True, exist_ok=True)
                self.today_path.write_text(json.dumps(live_today, indent=2), encoding="utf-8")
            except Exception:
                # If live recompute fails, keep the last saved payload.
                pass
            pred = pd.read_parquet(self.pred_path) if self.pred_path.exists() else pd.DataFrame()
            labels = pd.read_parquet(self.labels_path) if self.labels_path.exists() else pd.DataFrame()
            feats = pd.read_parquet(self.features_path) if self.features_path.exists() else pd.DataFrame()
            return {"today": today, "pred": pred, "labels": labels, "features": feats}

        self.run_in_bg("Refresh Data", task, self._apply_loaded_data)

    def _apply_loaded_data(self, payload: dict[str, Any]) -> None:
        self.today_signal = payload["today"]
        self.recent_df = payload["pred"].copy()
        self.stats_df = self._compute_quick_stats(payload["pred"], payload["labels"], payload["features"])

        self.meta_var.set(
            f"session: {self.today_signal.get('session_id', '--')} | prediction_ts_et: {self.today_signal.get('prediction_ts_et', '--')}"
        )
        pbull = self.today_signal.get("p_bull")
        pbear = self.today_signal.get("p_bear")
        conf = abs((pbull or 0.5) - 0.5) * 2 if pbull is not None else None
        self.pbull_var.set(f"p_bull: {pbull:.4f}" if pbull is not None else "p_bull: --")
        self.pbear_var.set(f"p_bear: {pbear:.4f}" if pbear is not None else "p_bear: --")
        self.conf_var.set(f"confidence: {conf:.3f}" if conf is not None else "confidence: --")
        self.freshness_var.set(f"freshness: {self.freshness(self.today_signal.get('prediction_ts_et'))}")

        self.recompute_bias()
        self.apply_filters()
        self._render_stats()

    def recompute_bias(self) -> None:
        pbull = self.today_signal.get("p_bull")
        if pbull is None:
            self.bias_var.set("--")
            return
        bias = apply_bias(float(pbull), float(self.th_long.get()), float(self.th_short.get()))
        self.bias_var.set(bias)

    def apply_filters(self) -> None:
        for row in self.tree.get_children():
            self.tree.delete(row)
        if self.recent_df.empty:
            return
        df = self.recent_df.sort_values("start_ts_et", ascending=False).head(max(1, int(self.last_n.get()))).copy()
        if self.filter_bias.get() != "ALL":
            df = df[df["bias"] == self.filter_bias.get()]
        for _, r in df.iterrows():
            self.tree.insert("", "end", values=(r["session_id"], str(r["start_ts_et"]), f"{r['p_bull']:.4f}", f"{r['p_bear']:.4f}", r["bias"]))

    def replay_selected(self) -> None:
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("Replay", "Select a session row first.")
            return
        item = self.tree.item(selected[0])
        session_id = item["values"][0]
        self.set_status(f"Replay target selected: {session_id}. Launching replay app.")
        self.open_replay()

    def open_replay(self) -> None:
        cmd = ["python", "-m", "src.replay_gui", "--bars", "data/processed/bars_with_session.parquet"]
        subprocess.Popen(cmd, cwd=self.root_dir)

    def run_pipeline(self) -> None:
        def task() -> None:
            commands = [
                ["python", "-m", "src.sessionize"],
                ["python", "-m", "src.labels"],
                ["python", "-m", "src.features"],
            ]
            if self.include_train.get():
                commands.append(["python", "-m", "src.train"])
            commands.extend(
                [
                    ["python", "-m", "src.predict"],
                    ["python", "-m", "src.evaluate"],
                    ["python", "-m", "src.monitor"],
                    ["python", "-m", "src.run_today"],
                ]
            )
            for cmd in commands:
                proc = subprocess.run(cmd, cwd=self.root_dir, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise RuntimeError(f"{' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")

        self.run_in_bg("Run Full Pipeline", task, lambda _: self.refresh_data())

    def _compute_quick_stats(self, pred: pd.DataFrame, labels: pd.DataFrame, features: pd.DataFrame) -> dict[str, Any]:
        stats = {
            "precision": None,
            "coverage": None,
            "bucket_winrates": {},
            "drift_warnings": [],
        }
        if pred.empty:
            return stats

        df = pred.copy()
        traded = df[df["bias"].isin(["LONG", "SHORT"])]
        stats["coverage"] = float(len(traded) / len(df)) if len(df) else None

        if not labels.empty and "y" in labels.columns:
            merged = df.merge(labels[["session_id", "y"]], on="session_id", how="inner")
            traded_m = merged[merged["bias"].isin(["LONG", "SHORT"])].copy()
            if not traded_m.empty:
                traded_m["correct"] = ((traded_m["bias"] == "LONG") & (traded_m["y"] == 1)) | (
                    (traded_m["bias"] == "SHORT") & (traded_m["y"] == 0)
                )
                stats["precision"] = float(traded_m["correct"].mean())
                traded_m["bucket"] = pd.cut(traded_m["p_bull"], bins=[0, 0.4, 0.5, 0.6, 1.0], include_lowest=True)
                bucket = traded_m.groupby("bucket", observed=False)["correct"].mean().dropna()
                stats["bucket_winrates"] = {str(k): float(v) for k, v in bucket.items()}

        if not features.empty:
            numeric = [c for c in features.columns if features[c].dtype.kind in "fi" and c != "y"]
            if numeric:
                split = int(len(features) * 0.7)
                train = features.iloc[:split]
                recent = features.iloc[max(split, len(features) - 50) :]
                for col in numeric[:8]:
                    denom = train[col].std() or 1.0
                    z_shift = (recent[col].mean() - train[col].mean()) / denom
                    if abs(z_shift) > 1.5:
                        stats["drift_warnings"].append(f"{col}: z_shift={z_shift:.2f}")
        return stats

    def _render_stats(self) -> None:
        self.stats_text.delete("1.0", tk.END)
        s = self.stats_df
        lines = [
            f"precision: {s['precision']:.3f}" if s["precision"] is not None else "precision: --",
            f"coverage: {s['coverage']:.3f}" if s["coverage"] is not None else "coverage: --",
            "bucket win-rates:",
        ]
        if s["bucket_winrates"]:
            for bucket, win in s["bucket_winrates"].items():
                lines.append(f"  {bucket}: {win:.3f}")
        else:
            lines.append("  --")
        lines.append("drift warnings:")
        if s["drift_warnings"]:
            lines.extend([f"  - {w}" for w in s["drift_warnings"]])
        else:
            lines.append("  none")
        self.stats_text.insert("1.0", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    args = parser.parse_args()
    app = AppUI(args.root)
    app.mainloop()


if __name__ == "__main__":
    main()
