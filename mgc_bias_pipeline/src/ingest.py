from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.io import read_table
from src.utils.logging import get_logger
from src.utils.time import to_et
from src.utils.validation import clean_ohlc, normalize_ohlc_columns

logger = get_logger(__name__)


def ingest(input_path: str, out_path: str, symbol: str = "MGC", timezone: str = "America/New_York") -> None:
    df = read_table(input_path)
    df = normalize_ohlc_columns(df)
    df["symbol"] = symbol
    df["ts_et"] = to_et(df["ts_raw"], timezone)
    df = clean_ohlc(df)
    df = df.sort_values("ts_et").drop_duplicates(subset=["ts_et", "symbol"]).reset_index(drop=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Ingested %d bars to %s", len(df), out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--symbol", default="MGC")
    p.add_argument("--out", default="data/raw/mgc_bars.parquet")
    args = p.parse_args()
    ingest(args.input, args.out, args.symbol)


if __name__ == "__main__":
    main()
