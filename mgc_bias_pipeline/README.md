# mgc_bias_pipeline

Production-style, config-driven pipeline for CME Micro Gold (MGC) NY-session bias prediction.

## Setup

```bash
cd mgc_bias_pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ required. Tkinter is part of stdlib, but on some Linux distros you may need OS package `python3-tk`.

## Expected input schema

CSV or Parquet with columns:
- timestamp column: one of `ts`, `timestamp`, `datetime`
- OHLC columns: `open`, `high`, `low`, `close`
- optional: `volume`

Timestamps may be naive or tz-aware. Naive timestamps are treated as UTC then converted to ET (`America/New_York`) in `ts_et`.

## Pipeline run order

```bash
python -m src.ingest --input /path/to/mgc.csv --out data/raw/mgc_bars.parquet
# or pull directly from an endpoint
python -m src.ingest --endpoint https://api.example.com/mgc/bars --endpoint-format json --records-path data.bars --query symbol=MGC --query timeframe=5m --header Authorization=Bearer\ <token> --out data/raw/mgc_bars.parquet
python -m src.sessionize --input data/raw/mgc_bars.parquet
python -m src.labels --input data/processed/bars_with_session.parquet
python -m src.features --bars data/processed/bars_with_session.parquet --labels data/processed/session_labels.parquet
python -m src.train --features data/processed/features_after_k.parquet
python -m src.predict --features data/processed/features_after_k.parquet --model_path data/outputs/model.joblib
python -m src.evaluate --pred data/outputs/predictions.parquet --labels data/processed/session_labels.parquet
python -m src.monitor --pred data/outputs/predictions.parquet --labels data/processed/session_labels.parquet --features data/processed/features_after_k.parquet
```

Artifacts produced:
- `data/processed/bars_with_session.parquet`
- `data/processed/session_labels.parquet`
- `data/processed/features_at_open.parquet`
- `data/processed/features_after_k.parquet`
- `data/outputs/model.joblib`
- `data/outputs/predictions.parquet`
- `data/outputs/eval_buckets.csv`
- `data/outputs/monitor_report.csv`

## Replay GUI

```bash
python -m src.replay_gui --bars data/processed/bars_with_session.parquet
```

GUI controls: session dropdown, Play/Pause, Step, Reset, speed slider, and scrub slider.
Candles are rendered with matplotlib patches (`Rectangle`) and wick lines.

## Control Center UI (`app_ui`)

```bash
python -m src.app_ui
```

The UI is designed to run fully offline against local files in this repo (no network dependency required at runtime).

Required/preferred local artifacts:
- required: `data/raw/mgc_bars.parquet`
- preferred: `data/processed/bars_with_session.parquet`
- preferred: `data/processed/features_after_k.parquet`
- required for model inference: `data/outputs/model.joblib`
- optional: `data/outputs/predictions.parquet`

Startup behavior:
- if `data/outputs/model.joblib` is missing, the signal area will indicate: **"Train model first."**
- if current session bars are insufficient for first-K features, the signal area will indicate: **"not enough bars yet"**

### Troubleshooting

- **Missing processed parquet files** (`bars_with_session.parquet`, `features_after_k.parquet`):
  run the minimal preprocessing path:

  ```bash
  python -m src.sessionize --input data/raw/mgc_bars.parquet
  python -m src.labels --input data/processed/bars_with_session.parquet
  python -m src.features --bars data/processed/bars_with_session.parquet --labels data/processed/session_labels.parquet
  ```

- **Need predictions table populated in the UI**:
  run minimal predict steps:

  ```bash
  python -m src.train --features data/processed/features_after_k.parquet
  python -m src.predict --features data/processed/features_after_k.parquet --model_path data/outputs/model.joblib
  ```

## Today signal

```bash
python -m src.run_today --bars data/raw/mgc_bars.parquet --model_path data/outputs/model.joblib
```

Writes `data/outputs/today_signal.json` and prints JSON:
```json
{"session_id": "NY_2024-01-20", "prediction_ts_et": "...", "p_bull": 0.61, "p_bear": 0.39, "bias": "LONG"}
```

## Leakage and timezone notes

- `features_at_open` uses only prior sessions (`shift(1)` and prior rolling windows).
- `features_after_k` uses bars where `ts_et < session_start + K minutes` only.
- Session assignment is deterministic and handles cross-midnight windows.

## Endpoint ingestion (no manual file prep)

`src.ingest` can now pull data directly from an HTTP endpoint and normalize common field aliases automatically:

- timestamp aliases: `ts`, `timestamp`, `datetime`, `time`, `date`, `t`
- OHLC aliases: `open/o`, `high/h`, `low/l`, `close/c`

Example JSON payload shapes supported:

- top-level list: `[{...}, {...}]`
- nested list/object via `--records-path` (e.g. `data.bars`)

CLI flags:

- `--endpoint`: URL to fetch
- `--endpoint-format`: `json` (default), `csv`, `parquet`
- `--records-path`: dot-path within JSON response
- `--query key=value`: repeatable query params
- `--header key=value`: repeatable request headers

This lets you run the full pipeline without manually shaping CSV/Parquet files first.
