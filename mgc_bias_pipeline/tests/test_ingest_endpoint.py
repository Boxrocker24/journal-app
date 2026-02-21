import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import pytest

from src.ingest import _prepare_bars, ingest
from src.utils.fetch import fetch_endpoint_table


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = b'{"data":{"bars":[{"t":"2024-01-01T00:00:00Z","o":100,"h":101,"l":99,"c":100.5}]}}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


def _start_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_fetch_endpoint_json_records_path():
    server = _start_server()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/bars"
        df = fetch_endpoint_table(endpoint, response_format="json", records_path="data.bars")
        assert {"t", "o", "h", "l", "c"}.issubset(df.columns)
        prepared = _prepare_bars(df, symbol="MGC", timezone="America/New_York")
        assert {"open", "high", "low", "close", "ts_et", "symbol"}.issubset(prepared.columns)
        assert len(prepared) == 1
    finally:
        server.shutdown()
        server.server_close()


def test_ingest_requires_exactly_one_source(tmp_path):
    out = tmp_path / "bars.parquet"
    with pytest.raises(ValueError, match="exactly one source"):
        ingest(out_path=str(out), input_path="a.csv", endpoint="http://example.com")

    with pytest.raises(ValueError, match="exactly one source"):
        ingest(out_path=str(out))


def test_prepare_bars_accepts_alias_columns():
    df = pd.DataFrame(
        {"time": ["2024-01-01T00:00:00Z"], "o": [100], "h": [101], "l": [99], "c": [100.5]}
    )
    out = _prepare_bars(df, symbol="MGC", timezone="America/New_York")
    assert out.loc[0, "open"] == 100
    assert out.loc[0, "close"] == 100.5
