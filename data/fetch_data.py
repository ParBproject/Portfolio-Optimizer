"""
fetch_data.py
-------------
Standalone script to download and cache historical price data using yfinance.
Run directly: python data/fetch_data.py
"""

import os
import pandas as pd
import yfinance as yf

# ── Default configuration ──────────────────────────────────────────────────────
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "SPY"]
DEFAULT_START   = "2015-01-01"
DEFAULT_END     = "2024-12-31"
CACHE_DIR       = os.path.join(os.path.dirname(__file__), "cache")


def download_prices(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted close prices for *tickers* between *start* and *end*.

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
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{'_'.join(sorted(tickers))}_{start}_{end}.csv")

    if cache and os.path.exists(cache_path):
        print(f"[cache] Loading prices from {cache_path}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    print(f"[yfinance] Downloading {tickers} from {start} to {end} …")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance returns a MultiIndex when >1 ticker; grab "Close" level
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"][tickers]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Forward-fill small gaps (e.g. staggered exchange holidays), then drop NaNs
    prices = prices.ffill().dropna()

    if cache:
        prices.to_csv(cache_path)
        print(f"[cache] Saved to {cache_path}")

    return prices


if __name__ == "__main__":
    df = download_prices()
    print(df.tail())
    print(f"\nShape: {df.shape}  |  Date range: {df.index[0].date()} → {df.index[-1].date()}")
