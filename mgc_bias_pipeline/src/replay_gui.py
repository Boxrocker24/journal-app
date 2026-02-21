from __future__ import annotations

import argparse
import tkinter as tk
from tkinter import ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


class ReplayApp(tk.Tk):
    def __init__(self, bars: pd.DataFrame, initial_session_id: str | None = None):
        super().__init__()
        self.title("MGC Session Replay")
        self.geometry("1200x700")
        self.bars = bars.sort_values("ts_et")
        self.meta = self._session_meta()
        self.current_idx = 0
        self.playing = False
        self.speed_ms = tk.IntVar(value=200)

        session_id_values = self.meta["session_id"].astype(str)
        default_session = session_id_values.iloc[0]
        requested_session = str(initial_session_id) if initial_session_id is not None else None
        selected_session = requested_session if requested_session in set(session_id_values.tolist()) else default_session

        self.session_var = tk.StringVar(value=selected_session)
        top = ttk.Frame(self)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Session:").pack(side=tk.LEFT)
        combo = ttk.Combobox(top, textvariable=self.session_var, values=self.meta["display"].tolist(), width=90)
        combo.pack(side=tk.LEFT, padx=5)
        combo.bind("<<ComboboxSelected>>", lambda e: self.reset())

        ttk.Button(top, text="Play/Pause", command=self.toggle_play).pack(side=tk.LEFT)
        ttk.Button(top, text="Step", command=self.step).pack(side=tk.LEFT)
        ttk.Button(top, text="Reset", command=self.reset).pack(side=tk.LEFT)
        ttk.Scale(top, from_=50, to=1000, variable=self.speed_ms, orient="horizontal").pack(side=tk.LEFT, padx=8)

        self.slider = ttk.Scale(self, from_=0, to=10, orient="horizontal", command=self.scrub)
        self.slider.pack(fill=tk.X)

        self.fig = Figure(figsize=(12, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.reset()

    def _session_meta(self) -> pd.DataFrame:
        g = self.bars.groupby("session_id", as_index=False).agg(
            session_name=("session_name", "first"),
            start=("ts_et", "min"),
            open=("open", "first"),
            close=("close", "last"),
            bars=("close", "size"),
        )
        g["ret"] = (g["close"] - g["open"]) / g["open"]
        g["display"] = g.apply(lambda r: f"{r.session_id} | {r.session_name} | ret={r.ret:.4f} | bars={r.bars}", axis=1)
        return g.sort_values("start")

    def get_session_data(self) -> pd.DataFrame:
        sess_id = self.session_var.get().split(" | ")[0]
        d = self.bars[self.bars["session_id"] == sess_id].reset_index(drop=True)
        self.slider.configure(to=max(len(d) - 1, 1))
        return d

    def draw(self):
        d = self.get_session_data().iloc[: self.current_idx + 1]
        self.ax.clear()
        for i, r in d.iterrows():
            color = "green" if r["close"] >= r["open"] else "red"
            self.ax.vlines(i, r["low"], r["high"], colors=color, linewidth=1)
            body_low = min(r["open"], r["close"])
            body_h = max(abs(r["close"] - r["open"]), 1e-6)
            self.ax.add_patch(Rectangle((i - 0.3, body_low), 0.6, body_h, facecolor=color, edgecolor=color))
        self.ax.set_title(self.session_var.get())
        self.ax.set_xlim(-1, max(len(d) + 1, 10))
        self.canvas.draw_idle()

    def step(self):
        d = self.get_session_data()
        self.current_idx = min(self.current_idx + 1, len(d) - 1)
        self.slider.set(self.current_idx)
        self.draw()

    def reset(self):
        self.current_idx = 0
        self.slider.set(0)
        self.draw()

    def scrub(self, value):
        self.current_idx = int(float(value))
        self.draw()

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self._tick()

    def _tick(self):
        if not self.playing:
            return
        d = self.get_session_data()
        if self.current_idx < len(d) - 1:
            self.step()
            self.after(self.speed_ms.get(), self._tick)
        else:
            self.playing = False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bars", default="data/processed/bars_with_session.parquet")
    p.add_argument("--session-id", default=None)
    args = p.parse_args()
    bars = pd.read_parquet(args.bars)
    app = ReplayApp(bars, initial_session_id=args.session_id)
    app.mainloop()


if __name__ == "__main__":
    main()
