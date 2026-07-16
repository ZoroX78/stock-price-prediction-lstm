import pandas as pd
import pandas_ta as ta
from app.services.data_service import data_service

class IndicatorService:
    """Computes RSI, MACD, Bollinger, ATR, etc."""

    async def compute_all(self, symbol: str) -> dict:
        df = await data_service.get_ohlcv(symbol)
        close = df["Close"]
        high, low = df["High"], df["Low"]

        rsi = ta.rsi(close, length=14)
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        bb = ta.bbands(close, length=20, std=2)
        atr = ta.atr(high, low, close, length=14)
        vol20 = close.pct_change().rolling(20).std()

        return {
            "rsi": round(float(rsi.iloc[-1]), 2),
            "macd": round(float(macd["MACD_12_26_9"].iloc[-1]), 4),
            "macd_signal": round(float(macd["MACDs_12_26_9"].iloc[-1]), 4),
            "macd_hist": round(float(macd["MACDh_12_26_9"].iloc[-1]), 4),
            "bb_upper": round(float(bb["BBU_20_2.0"].iloc[-1]), 2),
            "bb_lower": round(float(bb["BBL_20_2.0"].iloc[-1]), 2),
            "atr": round(float(atr.iloc[-1]), 2),
            "rolling_20d_vol": round(float(vol20.iloc[-1]), 4),
            "series": {
                "rsi": rsi.dropna().round(2).tolist(),
                "macd": macd["MACD_12_26_9"].dropna().round(4).tolist(),
                "macd_signal": macd["MACDs_12_26_9"].dropna().round(4).tolist(),
            },
        }

indicator_service = IndicatorService()