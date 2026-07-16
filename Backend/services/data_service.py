import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.core.config import settings
from app.core.cache import redis_cache
import json

class DataService:
    """Fetches and caches 60-day OHLCV data via yfinance."""

    async def get_ohlcv(self, symbol: str, days: int = None) -> pd.DataFrame:
        days = days or settings.LOOKBACK_DAYS
        cache_key = f"ohlcv:{symbol}:{days}"
        cached = await redis_cache.get(cache_key)
        if cached:
            return pd.read_json(cached, orient="split")

        end = datetime.now()
        start = end - timedelta(days=days + 15)  # buffer for weekends
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, auto_adjust=False)
        df = df.tail(days).copy()

        if df.empty:
            raise ValueError(f"No data for {symbol}")

        df.reset_index(inplace=True)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        
        await redis_cache.set(cache_key, df.to_json(orient="split"),
                              ttl=settings.CACHE_TTL_DATA)
        return df

    async def get_latest_price(self, symbol: str) -> dict:
        df = await self.get_ohlcv(symbol)
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        change = float(last["Close"] - prev["Close"])
        pct = (change / float(prev["Close"])) * 100 if prev["Close"] else 0
        return {
            "symbol": symbol,
            "price": round(float(last["Close"]), 2),
            "change": round(change, 2),
            "change_pct": round(pct, 2),
            "volume": int(last["Volume"]),
            "date": last["Date"],
        }

data_service = DataService()