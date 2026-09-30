"""Download the default ticker universe and write data/snapshot.

The browser demo cannot call Yahoo Finance, so this CSV is what it optimises.
Local ``streamlit run app.py`` does not read the snapshot unless
``PORTFOLIO_PRICE_SOURCE=snapshot``.

    python scripts/refresh_snapshot.py

``end`` is exclusive, matching yfinance. 2025-01-01 keeps the last session
of 2024, which is the app's default window.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.fetch_data import DEFAULT_START, DEFAULT_TICKERS, download_prices

# fetch_data.DEFAULT_END is "2024-12-31", and yfinance treats end as
# exclusive, so a download that stops there drops 2024-12-31. Ask through
# the next day so the snapshot covers the app's default window.
REQUEST_START = DEFAULT_START
REQUEST_END_EXCLUSIVE = "2025-01-01"
OUT_DIR = ROOT / "data" / "snapshot"


def main() -> None:
    os.environ["PORTFOLIO_PRICE_SOURCE"] = "yahoo"
    prices = download_prices(
        list(DEFAULT_TICKERS),
        start=REQUEST_START,
        end=REQUEST_END_EXCLUSIVE,
        cache=False,
    )
    index = prices.index
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    prices = prices.copy()
    prices.index = index.normalize()
    prices.index.name = "Date"
    prices = prices.sort_index()
    if prices.empty:
        raise RuntimeError("Yahoo Finance returned no rows for the snapshot.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices_path = OUT_DIR / "prices.csv"
    prices.to_csv(prices_path, date_format="%Y-%m-%d")
    meta = {
        "source": "Yahoo Finance",
        "adjustment": "split- and dividend-adjusted Close (yfinance auto_adjust=True)",
        "tickers": list(prices.columns),
        "request_start": REQUEST_START,
        "request_end_exclusive": REQUEST_END_EXCLUSIVE,
        "first_session": prices.index[0].strftime("%Y-%m-%d"),
        "last_session": prices.index[-1].strftime("%Y-%m-%d"),
        "downloaded_on": date.today().isoformat(),
        "rows": int(len(prices)),
        "note": (
            "Static snapshot for the in-browser demo. "
            "Local runs download live prices from Yahoo Finance."
        ),
    }
    (OUT_DIR / "snapshot.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {prices_path} ({meta['rows']} rows, "
        f"{meta['first_session']} → {meta['last_session']})"
    )


if __name__ == "__main__":
    main()
