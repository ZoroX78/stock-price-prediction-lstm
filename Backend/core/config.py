from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    APP_NAME: str = "Stock Analysis API"
    API_V1_PREFIX: str = "/api/v1"
    
    # Data
    LOOKBACK_DAYS: int = 60           # yfinance scrape window
    PREDICTION_HORIZON: int = 5       # next 5 days
    LSTM_SEQ_LEN: int = 30            # sliding window for LSTM
    TFT_SEQ_LEN: int = 30
    
    # Cache
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_DATA: int = 300         # 5 min for OHLCV
    CACHE_TTL_PRED: int = 600         # 10 min for predictions
    
    # Models
    MODEL_DIR: Path = Path("app/models/weights")
    
    # Watchlist (matches UI screenshot)
    WATCHLIST: list[str] = ["AAPL", "MSFT", "TSLA", "NVDA"]
    
    class Config:
        env_file = ".env"

settings = Settings()