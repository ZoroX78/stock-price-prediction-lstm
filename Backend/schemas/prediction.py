from pydantic import BaseModel
from typing import List

class PredictionResponse(BaseModel):
    symbol: str
    horizon_days: int
    last_price: float
    dates: List[str]
    lstm: List[float]
    tft: List[float]
    ensemble: List[float]
    confidence: float

class SignalResponse(BaseModel):
    symbol: str
    signal: str  # STRONG UP | UP | NEUTRAL | DOWN | STRONG DOWN
    composite_score: float
    confidence: float
    expected_return_5d: float
    target_price: float