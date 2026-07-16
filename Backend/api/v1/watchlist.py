from fastapi import APIRouter
from app.core.config import settings
from app.services.signal_service import signal_service
from app.services.data_service import data_service
from app.services.model_service import model_service

router = APIRouter()

@router.get("/")
async def get_watchlist():
    results = []
    for sym in settings.WATCHLIST:
        try:
            model_service.load(sym)
            sig = await signal_service.generate(sym)
            quote = await data_service.get_latest_price(sym)
            results.append({
                "symbol": sym,
                "price": quote["price"],
                "change_pct": quote["change_pct"],
                "signal": sig["signal"],
                "confidence": sig["confidence"],
            })
        except Exception:
            results.append({"symbol": sym, "error": "model_not_loaded"})
    return {"watchlist": results}