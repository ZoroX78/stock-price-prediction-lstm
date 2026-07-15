"""Unit tests for feature engineering and label construction modules."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from src.data.features import compute_features
from src.data.labels import compute_target_labels


@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    """Generate dummy stock price data for testing."""
    np.random.seed(42)
    n_days = 100
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="D")
    
    # Generate geometric Brownian motion-like paths
    close = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, n_days)))
    high = close * (1.0 + np.abs(np.random.normal(0.005, 0.002, n_days)))
    low = close * (1.0 - np.abs(np.random.normal(0.005, 0.002, n_days)))
    open_p = (high + low) / 2.0
    volume = np.random.randint(10000, 1000000, n_days).astype(float)
    
    df = pd.DataFrame(
        {
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    return df


def test_target_labels(sample_market_data):
    """Test that target labels are computed correctly and have correct shapes."""
    horizon = 5
    labels = compute_target_labels(sample_market_data, horizon=horizon)
    
    assert len(labels) == len(sample_market_data)
    assert isinstance(labels, pd.Series)
    
    # Check that the last `horizon` elements are NaN (no future data)
    assert labels.iloc[-horizon:].isna().all()
    # Check that previous elements are NOT NaN
    assert not labels.iloc[:-horizon].isna().any()
    
    # Check binary values
    valid_labels = labels.dropna()
    assert set(valid_labels.unique()).issubset({0.0, 1.0})
    
    # Verify values manually for a specific day
    t = 10
    pct_change = (
        sample_market_data["Close"].iloc[t + horizon] - sample_market_data["Close"].iloc[t]
    ) / sample_market_data["Close"].iloc[t]
    expected = 1.0 if pct_change > 0 else 0.0
    assert labels.iloc[t] == expected


def test_compute_features(sample_market_data):
    """Test that compute_features executes and returns correct columns and shapes."""
    features = compute_features(sample_market_data)
    
    assert len(features) == len(sample_market_data)
    assert isinstance(features, pd.DataFrame)
    
    # Check key columns
    expected_cols = [
        "daily_return",
        "log_return",
        "rsi_14",
        "macd",
        "bb_position",
        "volume_sma_ratio",
        "atr_normalized",
        "day_sin",
        "day_cos",
    ]
    for col in expected_cols:
        assert col in features.columns
        
    # Check that there are NaN values in the warmup period (e.g., SMA-50 needs 50 days)
    # The first 49 days should have some NaNs for 50-day indicators
    assert features["sma_50_slope"].iloc[:49].isna().any()
    assert not features["sma_50_slope"].iloc[55:].isna().any()
