import os
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

# =========================================================
# 1. DEEP LEARNING MODEL DEFINITIONS
# =========================================================

class StockLSTMClassifier(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, num_layers=2, dropout_prob=0.2):
        super(StockLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout_prob if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (Batch, Sequence, Features)
        lstm_out, (hn, cn) = self.lstm(x)
        # Extract last time step (Day 60)
        last_time_step = lstm_out[:, -1, :]
        out = self.fc(self.dropout(last_time_step))
        return self.sigmoid(out).squeeze()

# =========================================================
# 2. FEATURE ENGINEERING
# =========================================================

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates technical indicators in pandas to match the training pipeline."""
    df = df.copy()
    df.sort_index(inplace=True)

    # 1. RSI (14 days)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = avg_loss.replace(0, 1e-10)  # Avoid division by zero
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # 2. MACD (12, 26, 9)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_12_26_9'] = ema_12 - ema_26
    df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
    df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']

    # 3. Bollinger Bands (20 days)
    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['BBL_20'] = sma_20 - (2 * std_20)
    df['BBM_20'] = sma_20
    df['BBU_20'] = sma_20 + (2 * std_20)

    # 4. EMAs (9 and 21)
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()

    return df

# =========================================================
# 3. SCALER RETRIEVAL & INFERENCE ENGINE
# =========================================================

class PredictionEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load LSTM Model
        self.lstm_model = StockLSTMClassifier(input_dim=14, hidden_dim=64, num_layers=2)
        lstm_path = "lstm_stock_model.pt"
        if os.path.exists(lstm_path):
            self.lstm_model.load_state_dict(torch.load(lstm_path, map_location=self.device, weights_only=True))
            print(f"✅ Loaded LSTM model from: {lstm_path}")
        else:
            print(f"⚠️ Warning: LSTM model file '{lstm_path}' not found.")
            
        self.lstm_model.to(self.device)
        self.lstm_model.eval()

        # Load TFT Model
        self.tft_model = None
        tft_path = os.path.join("outputs", "checkpoints", "tft_best.ckpt")
        if os.path.exists(tft_path):
            try:
                self.tft_model = TemporalFusionTransformer.load_from_checkpoint(tft_path, map_location=self.device)
                self.tft_model.to(self.device)
                self.tft_model.eval()
                print(f"✅ Loaded TFT model from: {tft_path}")
            except Exception as e:
                print(f"❌ Error loading TFT model: {e}")
        else:
            print(f"⚠️ Warning: TFT checkpoint file '{tft_path}' not found.")

    def get_scaler_for_ticker(self, ticker: str) -> MinMaxScaler:
        """Loads ticker-specific MinMaxScaler from offline dataset, or fits on downloaded history."""
        ticker = ticker.upper().strip()
        filename = f"{ticker}_processed.csv"
        processed_path = os.path.join("data", "processed", filename)
        
        features_cols = [
            'Close', 'High', 'Low', 'Open', 'Volume', 'RSI_14', 
            'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9', 
            'BBL_20', 'BBM_20', 'BBU_20', 'EMA_9', 'EMA_21'
        ]
        
        if os.path.exists(processed_path):
            try:
                df_hist = pd.read_csv(processed_path)
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(df_hist[features_cols].values)
                return scaler
            except Exception as e:
                print(f"⚠️ Error reading historical CSV for {ticker}: {e}. Falling back to dynamic scaler.")
        
        # Fallback: Download 5 years of historical data to fit scaler
        print(f"🔄 Fetching historical data to fit scaler dynamically for {ticker}...")
        df_raw = yf.download(ticker, period="5y", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)
            
        df_feats = calculate_technical_indicators(df_raw)
        df_feats.dropna(inplace=True)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(df_feats[features_cols].values)
        return scaler

    def fetch_recent_data(self, ticker: str) -> pd.DataFrame:
        """Fetches recent stock price history from yfinance."""
        ticker = ticker.upper().strip()
        # Fetch 6 months of daily data to ensure we have enough lookback window
        # after technical indicator warmups (about 120 trading days needed)
        df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No stock data found for ticker symbol: '{ticker}'")
            
        # Clean multi-index columns if returned
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        return df

    def predict_lstm(self, df_recent: pd.DataFrame, ticker: str) -> dict:
        """Runs LSTM sequence-to-one prediction on the latest 60 trading days."""
        # 1. Compute indicators
        df_feats = calculate_technical_indicators(df_recent)
        df_feats.dropna(inplace=True)
        
        if len(df_feats) < 60:
            raise ValueError(f"Insufficient trading data for {ticker} (got {len(df_feats)}, need at least 60 trading days after indicator warm-up)")
            
        # Select last 60 days
        df_window = df_feats.tail(60).copy()
        
        features_cols = [
            'Close', 'High', 'Low', 'Open', 'Volume', 'RSI_14', 
            'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9', 
            'BBL_20', 'BBM_20', 'BBU_20', 'EMA_9', 'EMA_21'
        ]
        
        # 2. Scale features
        scaler = self.get_scaler_for_ticker(ticker)
        scaled_values = scaler.transform(df_window[features_cols].values)
        
        # Convert to PyTorch Tensor
        x_input = torch.tensor(scaled_values, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 3. Perform Monte Carlo Dropout Inference
        self.lstm_model.eval()
        # Turn on dropout specifically
        for m in self.lstm_model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
                
        mc_samples = 50
        probs = []
        with torch.no_grad():
            for _ in range(mc_samples):
                prob = self.lstm_model(x_input)
                # handle scalar vs single-item array
                val = prob.item() if hasattr(prob, 'item') else float(prob)
                probs.append(val)
                
        probs = np.array(probs)
        mean_prob = float(probs.mean())
        std_prob = float(probs.std())
        
        # Reset model to standard eval mode
        self.lstm_model.eval()
        
        # Determine signal and confidence
        is_bullish = mean_prob >= 0.5
        if is_bullish:
            signal = "STRONG UP" if mean_prob > 0.65 and std_prob < 0.15 else "UP"
            confidence = mean_prob * 100
        else:
            signal = "STRONG DOWN" if mean_prob < 0.35 and std_prob < 0.15 else "DOWN"
            confidence = (1 - mean_prob) * 100
            
        # Ensure confidence is formatted properly (between 50% and 100%)
        confidence = float(np.clip(confidence, 50.0, 99.9))
        
        return {
            "mean_prob": mean_prob,
            "std_prob": std_prob,
            "signal": signal,
            "confidence": round(confidence, 1),
            "features_df": df_window
        }

    def predict_tft(self, df_recent: pd.DataFrame, ticker: str) -> dict:
        """Runs Temporal Fusion Transformer forecast for the next 5 days."""
        if self.tft_model is None:
            raise RuntimeError("Temporal Fusion Transformer model checkpoint is not loaded.")
            
        # 1. Compute indicators
        df_feats = calculate_technical_indicators(df_recent)
        df_feats.dropna(inplace=True)
        
        if len(df_feats) < 60:
            raise ValueError(f"Insufficient trading data for {ticker} (got {len(df_feats)}, need at least 60 trading days)")
            
        df_window = df_feats.tail(60).copy()
        
        # 2. Format DataFrame for TFT inference
        # Required features: Ticker, time_idx, Close, Volume, RSI_14, MACD_12_26_9, BBL_20, EMA_9, Close_Target
        df_tft = df_window[['Close', 'Volume', 'RSI_14', 'MACD_12_26_9', 'BBL_20', 'EMA_9']].copy()
        df_tft['Ticker'] = ticker
        df_tft['time_idx'] = np.arange(len(df_tft))
        df_tft['Close_Target'] = df_tft['Close']
        
        # Create future 5 prediction rows
        last_row = df_tft.iloc[-1].copy()
        future_rows = []
        last_time_idx = int(last_row['time_idx'])
        
        for i in range(1, 6):
            new_row = last_row.copy()
            new_row['time_idx'] = last_time_idx + i
            new_row['Close_Target'] = 0.0  # Unknown target
            future_rows.append(new_row)
            
        df_future = pd.DataFrame(future_rows)
        df_tft = pd.concat([df_tft, df_future], ignore_index=True)
        
        # 3. Create TimeSeriesDataSet and Predict
        # Reconstruct TimeSeriesDataSet using model's dataset parameters
        dataset_params = self.tft_model.dataset_parameters
        inference_ds = TimeSeriesDataSet(
            df_tft,
            predict=True,
            stop_randomization=True,
            **dataset_params
        )
        dataloader = inference_ds.to_dataloader(batch_size=1, shuffle=False)
        
        with torch.no_grad():
            preds = self.tft_model.predict(dataloader, mode="prediction")
            
        # Descaled point predictions
        pred_prices = preds[0].cpu().numpy().tolist()
        
        # Determine signal based on the predicted 5-day return
        current_price = float(last_row['Close'])
        predicted_final_price = float(pred_prices[-1])
        pct_change = ((predicted_final_price - current_price) / current_price) * 100
        
        if pct_change > 1.5:
            signal = "STRONG UP"
        elif pct_change > 0.0:
            signal = "UP"
        elif pct_change < -1.5:
            signal = "DOWN"
        else:
            signal = "NEUTRAL"
            
        # Confidence score derived from directional probability or mock-calibrated to prediction scale
        confidence = float(50.0 + min(abs(pct_change) * 10, 49.0))
        confidence = float(np.clip(confidence, 50.0, 99.0))
        
        return {
            "predictions": pred_prices,
            "signal": signal,
            "confidence": round(confidence, 1),
            "features_df": df_window
        }
