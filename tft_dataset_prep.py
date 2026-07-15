import os
import glob
import pandas as pd

def build_unified_tft_dataframe(processed_dir):
    print("🔄 Compiling all stock files into a unified TFT master frame...")
    search_path = os.path.join(processed_dir, "*.csv")
    csv_files = glob.glob(search_path)
    
    all_frames = []
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        ticker = filename.split('_')[0]
        
        df = pd.read_csv(file_path, parse_dates=True)
        # Add the ticker symbol as a static categorical column
        df['Ticker'] = ticker
        
        # Ensure it is sorted chronologically
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Create the continuous time index required by TFT
        df['time_idx'] = df.index
        
        all_frames.append(df)
        
    master_df = pd.concat(all_frames, ignore_index=True)
    
    # Clean up column spaces and ensure target is a float for quantile regression
    master_df['Close_Target'] = master_df['Close'] # We will forecast the actual close price
    
    print(f"📊 Master TFT Dataframe compiled successfully: {master_df.shape}")
    return master_df

if __name__ == "__main__":
    PROCESSED_DIR = "data/processed"
    master_df = build_unified_tft_dataframe(PROCESSED_DIR)
    master_df.to_csv("data/tft_master_dataset.csv", index=False)