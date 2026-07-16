"""Target label construction module.

This module computes the target label for the sequence-to-one binary classification
task, determining whether the closing price of a stock goes UP or DOWN over the
next 5 trading days.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def compute_target_labels(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Compute binary target labels (1 for UP, 0 for DOWN) over a future horizon.

    Args:
        df: DataFrame containing a 'Close' price column.
        horizon: Prediction horizon in trading days (default: 5).

    Returns:
        A Series of labels aligned with the input DataFrame's index.
        The last `horizon` elements will be NaN because future data is unavailable.
    """
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")

    # Calculate percentage change between Close at t + horizon and Close at t
    # shift(-horizon) shifts future values back to current row
    future_close = df["Close"].shift(-horizon)
    pct_change = (future_close - df["Close"]) / df["Close"]

    # Binary classification: 1 if UP (pct_change > 0), 0 if DOWN (pct_change <= 0)
    # Note: NaN values remain NaN (e.g., the last `horizon` rows)
    labels = (pct_change > 0).astype(float)
    
    # Preserve NaNs for rows where future data is not available
    labels[pct_change.isna()] = np.nan

    return labels
