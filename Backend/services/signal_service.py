from app.services.model_service import model_service
from app.services.indicator_service import indicator_service

class SignalService:
    """Combines model predictions + indicators → STRONG UP/UP/NEUTRAL/DOWN/STRONG DOWN."""

    async def generate(self, symbol: str) -> dict:
        pred = await model_service.predict(symbol)
        ind = await indicator_service.compute_all(symbol)

        last = pred["last_price"]
        target = pred["ensemble"][-1]
        expected_ret = (target - last) / last

        # Score components
        model_score = np.tanh(expected_ret * 10)              # [-1, 1]
        rsi_score = 1 if ind["rsi"] > 55 else (-1 if ind["rsi"] < 45 else 0)
        macd_score = 1 if ind["macd_hist"] > 0 else -1

        composite = 0.6 * model_score + 0.2 * rsi_score + 0.2 * macd_score
        conf = pred["confidence"]

        if composite > 0.4 and conf > 0.75:
            signal = "STRONG UP"
        elif composite > 0.15:
            signal = "UP"
        elif composite < -0.4 and conf > 0.75:
            signal = "STRONG DOWN"
        elif composite < -0.15:
            signal = "DOWN"
        else:
            signal = "NEUTRAL"

        return {
            "symbol": symbol,
            "signal": signal,
            "composite_score": round(float(composite), 3),
            "confidence": round(float(conf), 3),
            "expected_return_5d": round(float(expected_ret * 100), 2),
            "target_price": round(float(target), 2),
        }

import numpy as np
signal_service = SignalService()