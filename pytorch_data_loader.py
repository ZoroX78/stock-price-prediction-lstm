import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class MultiStockDataset(Dataset):
    def __init__(self, processed_dir, sequence_length=60, split_type='train'):
        self.sequence_length = sequence_length
        self.x_samples = []
        self.y_samples = []
        
        # Find all processed CSV files
        search_path = os.path.join(processed_dir, "*.csv")
        csv_files = glob.glob(search_path)
        
        print(f"📦 Loading datasets for {split_type.upper()} split...")
        
        for file_path in csv_files:
            # Read processed data
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            df.sort_index(inplace=True)
            
            # 1. APPLY CHRONOLOGICAL SPLITTING
            # Years 2016-2023: Training | Year 2024: Validation | Years 2025-2026: Testing
            if split_type == 'train':
                df = df[df.index.year <= 2023]
            elif split_type == 'val':
                df = df[df.index.year == 2024]
            elif split_type == 'test':
                df = df[df.index.year >= 2025]
                
            # If the split leaves the dataframe empty (or too short), skip this ticker
            if len(df) <= sequence_length:
                continue
                
            # 2. SEPARATE FEATURES AND TARGET
            features = df.drop(columns=['Target']).values
            targets = df['Target'].values
            
            # 3. SCALE FEATURES PER TICKER
            # Crucial: Fits the scaler specifically to this company's variance footprint
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_features = scaler.fit_transform(features)
            
            # 4. SLIDE WINDOWS SECURELY (No bleeding into other stocks)
            for i in range(len(scaled_features) - sequence_length):
                x_window = scaled_features[i : i + sequence_length]
                y_target = targets[i + sequence_length]
                
                self.x_samples.append(x_window)
                self.y_samples.append(y_target)
                
        # Convert lists into single, unified PyTorch Tensors
        self.x_tensor = torch.tensor(np.array(self.x_samples), dtype=torch.float32)
        self.y_tensor = torch.tensor(np.array(self.y_samples), dtype=torch.float32)
        
        print(f"✅ Created {len(self.x_tensor)} unique 3D sequences.")

    def __len__(self):
        return len(self.x_tensor)

    def __getitem__(self, idx):
        return self.x_tensor[idx], self.y_tensor[idx]

if __name__ == "__main__":
    PROCESSED_DATA_DIR = "data/processed"
    LOOKBACK_WINDOW = 60
    BATCH_SIZE = 64
    
    # Instantiate the independent train, validation, and test structures
    train_dataset = MultiStockDataset(PROCESSED_DATA_DIR, sequence_length=LOOKBACK_WINDOW, split_type='train')
    val_dataset   = MultiStockDataset(PROCESSED_DATA_DIR, sequence_length=LOOKBACK_WINDOW, split_type='val')
    test_dataset  = MultiStockDataset(PROCESSED_DATA_DIR, sequence_length=LOOKBACK_WINDOW, split_type='test')
    
    # Wrap them into DataLoaders for mini-batch training acceleration
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Verify the shapes
    sample_x, sample_y = next(iter(train_loader))
    print("\n--- Final Tensor Check ---")
    print(f"Batch Input Tensor Shape (X): {sample_x.shape} -> (Batch Size, Sequence Length, Features)")
    print(f"Batch Target Tensor Shape (y): {sample_y.shape} -> (Batch Size,)")