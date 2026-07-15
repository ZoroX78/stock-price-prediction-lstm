import os
import glob
import pandas as pd
import numpy as np

def engineer_features(file_path):
    """Performs technical analysis feature engineering on a single raw stock dataframe."""
    # Load data and set Date as index
    df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
    
    # Ensure chronological order
    df.sort_index(inplace=True)

    # ---------------------------------------------------------
    # 1. TECHNICAL INDICATORS (The Input Features)
    # ---------------------------------------------------------
    
    # RSI (14 days)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_12_26_9'] = ema_12 - ema_26
    df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
    df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']

    # Bollinger Bands (20 days)
    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['BBL_20'] = sma_20 - (2 * std_20)
    df['BBM_20'] = sma_20
    df['BBU_20'] = sma_20 + (2 * std_20)

    # EMAs (9 and 21)
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()

    # ---------------------------------------------------------
    # 2. TARGET GENERATION (5 days into the future)
    # ---------------------------------------------------------
    prediction_horizon = 5
    df['Future_Close'] = df['Close'].shift(-prediction_horizon)
    df['Target'] = (df['Future_Close'] > df['Close']).astype(int)

    # ---------------------------------------------------------
    # 3. CLEAN UP & PREVENT DATA LEAKAGE
    # ---------------------------------------------------------
    df.dropna(inplace=True)
    df.drop(columns=['Future_Close'], inplace=True)
    
    return df

def batch_process_pipeline(input_folder, output_folder):
    """Scans input folder for CSVs, runs feature engineering, and saves to output folder."""
    # Create the output directory if it doesn't exist yet
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 Created directory: {output_folder}")
        
    # Find all CSV files inside data/raw
    search_path = os.path.join(input_folder, "*.csv")
    csv_files = glob.glob(search_path)
    
    total_files = len(csv_files)
    print(f"🚀 Found {total_files} raw files to process in '{input_folder}'\n")
    
    if total_files == 0:
        print("⚠️ No CSV files found. Please check your folder structure path.")
        return

    success_count = 0
    
    # Loop through every file automatically
    for idx, file_path in enumerate(csv_files, 1):
        filename = os.path.basename(file_path)
        ticker = filename.split('_')[0] # Extracts company name (e.g., 'AAPL' from 'AAPL_10y.csv')
        
        print(f"⏳ [{idx}/{total_files}] Processing: {ticker}")
        
        try:
            # Run the feature engineering calculations
            processed_df = engineer_features(file_path)
            
            # Construct the new save path
            output_file_name = f"{ticker}_processed.csv"
            output_file_path = os.path.join(output_folder, output_file_name)
            
            # Save the file
            processed_df.to_csv(output_file_path)
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {ticker}: {str(e)}")
            continue

    print(f"\n🎉 Pipeline Execution Finished Complete!")
    print(f"📊 Successfully processed and saved [{success_count}/{total_files}] files to '{output_folder}'.")

if __name__ == "__main__":
    # Define your pipeline paths
    RAW_DATA_DIR = "data/raw"
    PROCESSED_DATA_DIR = "data/processed"
    
    # Run the automated batch loop
    batch_process_pipeline(RAW_DATA_DIR, PROCESSED_DATA_DIR)