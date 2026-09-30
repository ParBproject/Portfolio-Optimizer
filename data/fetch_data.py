"""
fetch_data.py
-------------
Standalone script to download and cache historical price data using yfinance.
Run directly: python data/fetch_data.py
"""

import json
import os
import sys

import pandas as pd

# ── Default configuration ──────────────────────────────────────────────────────
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "SPY"]
DEFAULT_START   = "2015-01-01"
DEFAULT_END     = "2024-12-31"
CACHE_DIR       = os.path.join(os.path.dirname(__file__), "cache")
SNAPSHOT_DIR    = os.path.join(os.path.dirname(__file__), "snapshot")
SNAPSHOT_PRICES = os.path.join(SNAPSHOT_DIR, "prices.csv")
SNAPSHOT_META   = os.path.join(SNAPSHOT_DIR, "snapshot.json")
# Holidays and long weekends only. A longer gap is left missing so a halt
# is not turned into a streak of zero returns.
MAX_FFILL_DAYS  = 5


def price_source() -> str:
    """``snapshot`` for the browser demo, ``yahoo`` for a normal local run.

    Pyodide cannot call Yahoo Finance (no ``yfinance``, and the browser would
    be blocked by CORS). ``PORTFOLIO_PRICE_SOURCE=snapshot`` or ``yahoo``
    overrides that detection so tests and a local preview can force either path.
    """
    forced = os.environ.get("PORTFOLIO_PRICE_SOURCE", "").strip().lower()
    if forced in {"snapshot", "yahoo"}:
        return forced
    if sys.platform == "emscripten":
        return "snapshot"
    return "yahoo"


def snapshot_metadata() -> dict:
    """Read the bundled snapshot's tickers and trading-day range."""
    if not os.path.exists(SNAPSHOT_META):
        raise ValueError(
            "The bundled price snapshot is missing its metadata. "
            "Run locally to download prices from Yahoo Finance."
        )
    with open(SNAPSHOT_META, encoding="utf-8") as handle:
        meta = json.load(handle)
    required = ("tickers", "first_session", "last_session", "downloaded_on")
    missing = [key for key in required if key not in meta]
    if missing:
        raise ValueError(f"Price snapshot metadata is missing {missing}.")
    return meta


def snapshot_label() -> str:
    """One sentence that names the snapshot and its actual date range."""
    meta = snapshot_metadata()
    tickers = [str(ticker) for ticker in meta["tickers"]]
    if len(tickers) <= 1:
        names = tickers[0] if tickers else "the bundled symbols"
    else:
        names = ", ".join(tickers[:-1]) + f", and {tickers[-1]}"
    return (
        "Static price snapshot — not live Yahoo Finance. "
        f"Split- and dividend-adjusted closes for {names}, "
        f"from {meta['first_session']} through {meta['last_session']} "
        f"(downloaded {meta['downloaded_on']}). "
        "Run this app locally for live Yahoo Finance prices."
    )


def load_snapshot(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Slice the bundled adjusted closes. ``end`` is exclusive, like yfinance."""
    if not os.path.exists(SNAPSHOT_PRICES):
        raise ValueError(
            "The bundled price snapshot is missing. "
            "Run locally to download prices from Yahoo Finance."
        )
    if not tickers:
        raise ValueError("Provide at least one ticker.")
    prices = pd.read_csv(SNAPSHOT_PRICES, index_col=0, parse_dates=True).sort_index()
    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        have = ", ".join(str(column) for column in prices.columns)
        raise ValueError(
            f"No snapshot prices for {missing}. "
            f"The bundled snapshot only includes {have}. "
            "Run the app locally to download other tickers from Yahoo Finance."
        )
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts <= start_ts:
        raise ValueError("End date must be after the start date.")
    window = prices.loc[
        (prices.index >= start_ts) & (prices.index < end_ts),
        list(tickers),
    ].dropna(how="any")
    if window.empty:
        meta = snapshot_metadata()
        raise ValueError(
            f"No snapshot rows from {start} up to {end} (end is exclusive). "
            "The static snapshot covers "
            f"{meta['first_session']} through {meta['last_session']}."
        )
    return window


def clean_prices(prices: pd.DataFrame, max_ffill: int = MAX_FFILL_DAYS) -> pd.DataFrame:
    """Forward-fill short gaps, then drop any row that is still incomplete.

    Filling is strictly backward-looking and stops after ``max_ffill`` sessions.
    """
    if prices.empty:
        raise ValueError("No price rows to clean.")
    ordered = prices.sort_index()
    entirely_missing = [str(col) for col in ordered.columns if ordered[col].isna().all()]
    if entirely_missing:
        raise ValueError(f"No prices for {entirely_missing}.")
    cleaned = ordered.ffill(limit=max_ffill).dropna(how="any")
    if cleaned.empty:
        raise ValueError(
            "No overlapping price history after cleaning gaps longer than "
            f"{max_ffill} sessions."
        )
    return cleaned


def extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Pull adjusted closes out of the frames yfinance has used over time.

    Handles a flat Close column, a (field, ticker) MultiIndex, and a
    (ticker, field) MultiIndex. Columns are returned in ``tickers`` order.
    """
    if raw is None or raw.empty:
        raise ValueError(f"No price data returned for {tickers}.")
    columns = raw.columns
    if isinstance(columns, pd.MultiIndex):
        level0 = set(columns.get_level_values(0))
        level1 = set(columns.get_level_values(1))
        if "Close" in level0:
            prices = raw["Close"]
        elif "Close" in level1:
            prices = raw.xs("Close", axis=1, level=1)
        else:
            raise ValueError("Downloaded data has no Close column.")
    else:
        if "Close" not in raw.columns:
            raise ValueError("Downloaded data has no Close column.")
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        raise ValueError(f"No prices for {missing}.")
    return clean_prices(prices.loc[:, tickers])


def download_prices(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted close prices for *tickers* between *start* and *end*.

    Local runs use Yahoo Finance. The in-browser demo
    (``sys.platform == "emscripten"``, or ``PORTFOLIO_PRICE_SOURCE=snapshot``)
    reads the bundled snapshot instead, because the browser cannot call Yahoo.

    With ``auto_adjust=True``, yfinance's Close column is split- and
    dividend-adjusted. ``end`` is exclusive, matching yfinance, including
    when the rows come from the snapshot.

    Parameters
    ----------
    tickers : list of str   Ticker symbols (e.g. ["AAPL", "MSFT"]).
    start   : str           ISO date string for the start of the period.
    end     : str           ISO date string for the end of the period.
    cache   : bool          If True, save/load a local CSV cache.

    Returns
    -------
    pd.DataFrame  Date-indexed DataFrame of adjusted close prices.
    """
    if price_source() == "snapshot":
        print("[snapshot] Using the bundled price snapshot (not live Yahoo Finance)")
        return load_snapshot(list(tickers), start, end)
    return _download_yahoo(list(tickers), start, end, cache)


def _download_yahoo(
    tickers: list[str],
    start: str,
    end: str,
    cache: bool,
) -> pd.DataFrame:
    """Live Yahoo Finance path used by local runs. Not available in the browser."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{'_'.join(sorted(tickers))}_{start}_{end}.csv")

    if cache and os.path.exists(cache_path):
        print(f"[cache] Loading prices from {cache_path}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    print(f"[yfinance] Downloading {tickers} from {start} to {end} …")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = extract_close(raw, list(tickers))

    if cache:
        prices.to_csv(cache_path)
        print(f"[cache] Saved to {cache_path}")

    return prices


if __name__ == "__main__":
    df = download_prices()
    print(df.tail())
    print(f"\nShape: {df.shape}  |  Date range: {df.index[0].date()} → {df.index[-1].date()}")
