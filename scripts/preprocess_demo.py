"""Demo script to test feature engineering on AAPL raw data."""

from __future__ import annotations
from pathlib import Path
import pandas as pd

from src.data.features import compute_features
from src.data.labels import compute_target_labels

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "AAPL_10y.csv"


def run_demo():
    print(f"Loading raw data from: {RAW_DATA_PATH}")
    if not RAW_DATA_PATH.exists():
        print(f"Error: {RAW_DATA_PATH} does not exist.")
        return

    # Load raw data
    df = pd.read_csv(RAW_DATA_PATH, index_col="Date", parse_dates=True)
    print(f"Loaded DataFrame with shape: {df.shape}")
    print("\nFirst 3 rows of raw data:")
    print(df.head(3))

    # Compute labels
    print("\nComputing target labels (5-day horizon)...")
    labels = compute_target_labels(df, horizon=5)
    
    # Compute features
    print("Computing technical features using TA-Lib...")
    try:
        features = compute_features(df)
    except Exception as e:
        print(f"Error computing features: {e}")
        return

    # Combine features and labels for analysis
    dataset = features.copy()
    dataset["Target_Label"] = labels

    print(f"\nEngineered dataset shape: {dataset.shape}")
    print(f"Total features created: {len(features.columns)}")
    print("\nFeature Columns:")
    print(list(features.columns))

    # Check for NaN values
    nans = dataset.isna().sum()
    print("\nNaN count per column (first 60 rows contain warm-up NaNs):")
    for col, count in nans.items():
        if count > 0:
            print(f"  - {col}: {count} NaNs")

    # Cleaned dataset (dropping warm-up and the last horizon elements that don't have labels)
    cleaned_dataset = dataset.dropna()
    print(f"\nCleaned dataset shape (after dropping NaNs): {cleaned_dataset.shape}")
    
    # Label class distribution
    class_counts = cleaned_dataset["Target_Label"].value_counts()
    class_pcts = cleaned_dataset["Target_Label"].value_counts(normalize=True) * 100
    print("\nLabel Class Distribution:")
    for cls, count in class_counts.items():
        label = "UP" if cls == 1 else "DOWN"
        print(f"  - Class {int(cls)} ({label}): {count} ({class_pcts[cls]:.2f}%)")


if __name__ == "__main__":
    run_demo()
