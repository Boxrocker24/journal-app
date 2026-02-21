from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.fetch import fetch_endpoint_table, parse_key_value_pairs
from src.utils.io import read_table
from src.utils.logging import get_logger
from src.utils.time import to_et
from src.utils.validation import clean_ohlc, normalize_ohlc_columns

logger = get_logger(__name__)


def ingest(
    out_path: str,
    symbol: str = "MGC",
    timezone: str = "America/New_York",
    input_path: str | None = None,
    endpoint: str | None = None,
    endpoint_format: str = "json",
    query: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    records_path: str | None = None,
) -> None:
    if bool(input_path) == bool(endpoint):
        raise ValueError("Provide exactly one source: --input or --endpoint")

    if input_path:
        df = read_table(input_path)
    else:
        df = fetch_endpoint_table(
            endpoint=endpoint or "",
            response_format=endpoint_format,
            query=query,
            headers=headers,
            records_path=records_path,
        )

    df = _prepare_bars(df, symbol=symbol, timezone=timezone)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Ingested %d bars to %s", len(df), out_path)


def _prepare_bars(df: pd.DataFrame, symbol: str, timezone: str) -> pd.DataFrame:
    df = normalize_ohlc_columns(df)
    df["symbol"] = symbol
    df["ts_et"] = to_et(df["ts_raw"], timezone)
    df = clean_ohlc(df)
    return df.sort_values("ts_et").drop_duplicates(subset=["ts_et", "symbol"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input")
    p.add_argument("--endpoint")
    p.add_argument("--endpoint-format", default="json", choices=["json", "csv", "parquet"])
    p.add_argument("--query", action="append", help="Query parameter in key=value format; repeatable")
    p.add_argument("--header", action="append", help="Header in key=value format; repeatable")
    p.add_argument("--records-path", default=None, help="Dot path to JSON array/object (e.g., data.bars)")
    p.add_argument("--symbol", default="MGC")
    p.add_argument("--out", default="data/raw/mgc_bars.parquet")
    args = p.parse_args()
    ingest(
        out_path=args.out,
        symbol=args.symbol,
        input_path=args.input,
        endpoint=args.endpoint,
        endpoint_format=args.endpoint_format,
        query=parse_key_value_pairs(args.query),
        headers=parse_key_value_pairs(args.header),
        records_path=args.records_path,
    )


if __name__ == "__main__":
    main()
