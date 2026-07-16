# In app/services/model_service.py
import torch
import joblib
from app.core.config import settings
from app.models.lstm_arch import LSTMModel
from app.models.tft_arch import TFTModel

class ModelService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = None
        self.tft = None
        self.scaler = None
        self.feature_cols = [
            "Close", "Volume", "RSI", "MACD",
            "MACD_Signal", "BB_Width", "ATR", "Return_1d"
        ]
        self.lstm_seq_len = settings.LSTM_SEQ_LEN

    def _clean_state_dict(self, checkpoint: dict) -> dict:
        """Strips PyTorch Lightning prefixes if present."""
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            
        # Remove 'model.' prefix if saved from a LightningModule wrapper
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    def load(self, symbol: str):
        """Loads weights for given symbol with error tolerance."""
        sym = symbol.lower()
        lstm_path = settings.MODEL_DIR / f"lstm_{sym}.pt"
        tft_path = settings.MODEL_DIR / f"tft_{sym}.pt"
        scaler_path = settings.MODEL_DIR / f"scaler_{sym}.pkl"

        if not lstm_path.exists() or not tft_path.exists():
            raise FileNotFoundError(f"Model weights for {symbol} not found in {settings.MODEL_DIR}")

        # 1. Initialize your architectures
        # IMPORTANT: Replace LSTMModel/TFTModel imports with your actual class definitions!
        self.lstm = LSTMModel(input_dim=len(self.feature_cols), output_dim=settings.PREDICTION_HORIZON)
        self.tft = TFTModel(input_dim=len(self.feature_cols), output_dim=settings.PREDICTION_HORIZON)

        # 2. Load Checkpoints safely
        lstm_ckpt = torch.load(lstm_path, map_location=self.device)
        tft_ckpt = torch.load(tft_path, map_location=self.device)

        lstm_state = self._clean_state_dict(lstm_ckpt)
        tft_state = self._clean_state_dict(tft_ckpt)

        # 3. Load state dicts (strict=False ignores minor architecture mismatches)
        try:
            self.lstm.load_state_dict(lstm_state, strict=True)
        except RuntimeError as e:
            print(f"Warning: Strict load failed for LSTM. Trying strict=False. Error: {e}")
            self.lstm.load_state_dict(lstm_state, strict=False)

        try:
            self.tft.load_state_dict(tft_state, strict=True)
        except RuntimeError as e:
            print(f"Warning: Strict load failed for TFT. Trying strict=False. Error: {e}")
            self.tft.load_state_dict(tft_state, strict=False)

        self.lstm.to(self.device).eval()
        self.tft.to(self.device).eval()

        # 4. Load Scaler
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        else:
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

# ... (keep the rest of the ModelService methods like _build_features and predict)