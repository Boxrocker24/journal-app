from __future__ import annotations

import json
from io import BytesIO, StringIO
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


class EndpointFetchError(RuntimeError):
    pass


def parse_key_value_pairs(pairs: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"Invalid key/value pair '{item}'. Expected format key=value")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _extract_records(payload: object, records_path: str | None) -> object:
    if not records_path:
        return payload
    cur = payload
    for part in records_path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise EndpointFetchError(f"records_path '{records_path}' not found in JSON payload")
    return cur


def fetch_endpoint_table(
    endpoint: str,
    response_format: str = "json",
    query: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    records_path: str | None = None,
    timeout_s: int = 30,
) -> pd.DataFrame:
    query = query or {}
    headers = headers or {}
    final_url = endpoint
    if query:
        final_url = f"{endpoint}{'&' if '?' in endpoint else '?'}{urlencode(query)}"

    req = Request(final_url, headers=headers)
    with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
        body = resp.read()

    fmt = response_format.lower()
    if fmt == "json":
        payload = json.loads(body.decode("utf-8"))
        records = _extract_records(payload, records_path)
        if isinstance(records, dict):
            return pd.DataFrame([records])
        if isinstance(records, list):
            return pd.DataFrame(records)
        raise EndpointFetchError("JSON payload must resolve to a dict or a list of dicts")
    if fmt == "csv":
        return pd.read_csv(StringIO(body.decode("utf-8")))
    if fmt == "parquet":
        return pd.read_parquet(BytesIO(body))

    raise ValueError("Unsupported response format; use json/csv/parquet")
