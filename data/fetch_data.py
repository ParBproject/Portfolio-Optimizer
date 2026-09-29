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
# Holidays and long weekends only. A longer gap is left missing so a halt
# is not turned into a streak of zero returns.
MAX_FFILL_DAYS  = 5


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

    With ``auto_adjust=True``, yfinance's Close column is split- and
    dividend-adjusted. ``end`` is exclusive, matching yfinance.

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
    prices = extract_close(raw, list(tickers))

    if cache:
        prices.to_csv(cache_path)
        print(f"[cache] Saved to {cache_path}")

    return prices


if __name__ == "__main__":
    df = download_prices()
    print(df.tail())
    print(f"\nShape: {df.shape}  |  Date range: {df.index[0].date()} → {df.index[-1].date()}")
