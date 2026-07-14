"""Download S&P 100 OHLCV history from Yahoo Finance.

Why the previous errors happened:
1. pd.read_html() needs an HTML parser (lxml) — pandas does not install it for you.
2. Wikipedia returns HTTP 403 for bare urllib requests (no browser-like User-Agent).
"""

from __future__ import annotations

import os
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# Resolve paths relative to this file so the script works from any cwd.
DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"

# Wikipedia blocks default Python User-Agents; use a normal browser string.
WIKI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def get_sp500_tickers() -> list[str]:
    """Scrape the current S&P 500 tickers from Wikipedia."""
    print("Fetching current S&P 500 list from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    # Fetch HTML with a proper User-Agent, then parse tables from the text.
    # Passing the URL directly to pd.read_html() uses urllib without headers → 403.
    response = requests.get(url, headers=WIKI_HEADERS, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    df = tables[0]

    # yfinance expects BRK.B-style symbols as BRK-B
    tickers = [str(t).replace(".", "-") for t in df["Symbol"].tolist()]

    print(f"Successfully loaded {len(tickers)} tickers.")
    return tickers


def download_market_data(tickers: list[str], years: int = 10, limit: int | None = 100) -> None:
    """Download historical data for tickers and save CSVs under data/raw/.

    Args:
        tickers: List of ticker symbols.
        years: Lookback period in years for yfinance.
        limit: If set, only download the first N tickers (useful for quick tests).
               Pass None to download all tickers.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    selected = tickers if limit is None else tickers[:limit]
    print(f"Starting download for {len(selected)} companies (of {len(tickers)})...")

    for ticker in selected:
        print(f"Fetching data for {ticker}...")
        try:
            data = yf.download(
                ticker,
                period=f"{years}y",
                interval="1d",
                progress=False,
                auto_adjust=True,
            )

            if not data.empty:
                # Flatten multi-level columns if yfinance returns them that way
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                file_path = RAW_DIR / f"{ticker}_{years}y.csv"
                data.to_csv(file_path)
                print(f"   Saved {ticker} to {file_path}")
            else:
                print(f"   No data found for {ticker}")

        except Exception as e:
            print(f"   Error fetching {ticker}: {e}")

        # Pause between requests to avoid rate limits from Yahoo Finance.
        time.sleep(1)

    print("Pipeline finished!")


if __name__ == "__main__":
    sp500_list = get_sp500_tickers()

    print("\nTop 10 companies loaded:")
    print(sp500_list[:10])
    print()

    # Default: first 5 tickers for a quick smoke test.
    # Use: limit=None  when you want all ~500.
    download_market_data(sp500_list, years=10, limit=100)
