"""Feature engineering module using TA-Lib and Pandas.

Constructs stationary, scale-invariant features from raw OHLCV market data.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import talib


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators and scale-invariant features from OHLCV data.

    Args:
        df: DataFrame containing columns: ['Open', 'High', 'Low', 'Close', 'Volume']
            and a DatetimeIndex (or a 'Date' column).

    Returns:
        DataFrame with computed features, aligned with the original index.
        Note: The first ~50 rows will contain NaNs due to warm-up periods for
        indicators like SMAs, MACD, and Bollinger Bands. These should be dropped
        during preprocessing rather than filled.
    """
    # Ensure columns exist and copy to avoid modifying original df
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain a '{col}' column.")

    features = pd.DataFrame(index=df.index)

    # 1. Base Prices & Volume (scaled/stationary)
    # Never use raw prices directly to avoid non-stationarity.
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_p = df["Open"].astype(float)
    volume = df["Volume"].astype(float)

    # Returns
    features["daily_return"] = close.pct_change()
    features["log_return"] = np.log(close / close.shift(1))
    features["open_close_ratio"] = (open_p - close.shift(1)) / close.shift(1)  # overnight gap
    features["high_low_ratio"] = (high - low) / close  # intraday range normalized

    # 2. RSI (Momentum) - 7, 14, 21 periods
    features["rsi_7"] = talib.RSI(close, timeperiod=7)
    features["rsi_14"] = talib.RSI(close, timeperiod=14)
    features["rsi_21"] = talib.RSI(close, timeperiod=21)

    # 3. MACD (Trend)
    macd, macd_signal, macd_hist = talib.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9
    )
    features["macd"] = macd
    features["macd_signal"] = macd_signal
    features["macd_hist"] = macd_hist

    # 4. Bollinger Bands (Volatility & Reversal)
    upper_band, middle_band, lower_band = talib.BBANDS(
        close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0
    )
    # Position: where Close sits relative to bands (0 to 1 normally)
    bb_range = upper_band - lower_band
    # Avoid divide-by-zero
    bb_range = bb_range.replace(0, np.nan)
    features["bb_position"] = (close - lower_band) / bb_range
    features["bb_width"] = bb_range / middle_band
    features["bb_signal"] = (close > middle_band).astype(float)

    # 5. Volume Momentum
    volume_sma_20 = talib.SMA(volume, timeperiod=20)
    volume_sma_20 = volume_sma_20.replace(0, np.nan)
    features["volume_sma_ratio"] = volume / volume_sma_20
    features["volume_change"] = volume.pct_change()
    features["obv"] = talib.OBV(close, volume)

    # 6. Volatility & Risk
    atr_14 = talib.ATR(high, low, close, timeperiod=14)
    features["atr_normalized"] = atr_14 / close
    features["realized_vol_20"] = features["log_return"].rolling(window=20).std()

    # 7. Stochastic Oscillator
    slowk, slowd = talib.STOCH(
        high,
        low,
        close,
        fastk_period=5,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0,
    )
    features["stoch_k"] = slowk
    features["stoch_d"] = slowd

    # 8. Rate of Change (ROC)
    features["roc_5"] = talib.ROC(close, timeperiod=5)
    features["roc_10"] = talib.ROC(close, timeperiod=10)
    features["roc_20"] = talib.ROC(close, timeperiod=20)

    # 9. Moving Average Slopes (Trend Acceleration)
    sma_10 = talib.SMA(close, timeperiod=10)
    sma_50 = talib.SMA(close, timeperiod=50)
    # Simple difference over 5 days to capture slope
    features["sma_10_slope"] = (sma_10 - sma_10.shift(5)) / close.shift(5)
    features["sma_50_slope"] = (sma_50 - sma_50.shift(5)) / close.shift(5)

    # 10. Calendar Features (Cyclical encoding of Day of Week)
    # Convert index or Date column to datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    if isinstance(df.index, pd.DatetimeIndex):
        day_of_week = df.index.dayofweek
        # Day of week ranges from 0 (Monday) to 4 (Friday) for trading days, or 6 for weekends
        # Max value is 6
        features["day_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
        features["day_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)
    else:
        # Fallback if no datetime index
        features["day_sin"] = 0.0
        features["day_cos"] = 0.0

    return features
