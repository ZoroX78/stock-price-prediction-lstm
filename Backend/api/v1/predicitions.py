from fastapi import APIRouter, HTTPException
from app.services.model_service import model_service
from app.services.signal_service import signal_service
from app.services.feature_service import feature_service
from app.services.validation_service import validation_service

router = APIRouter()

@router.post("/{symbol}/predict")
async def predict(symbol: str):
    sym = symbol.upper()
    try:
        model_service.load(sym)
        return await model_service.predict(sym)
    except FileNotFoundError:
        raise HTTPException(404, f"Model weights for {sym} not found")
    except Exception as e:
        raise HTTPException(500, str(e))

@router.get("/{symbol}/signal")
async def get_signal(symbol: str):
    model_service.load(symbol.upper())
    return await signal_service.generate(symbol.upper())

@router.get("/{symbol}/features")
async def get_features(symbol: str):
    model_service.load(symbol.upper())
    return await feature_service.compute_importance(symbol.upper())

@router.get("/{symbol}/validation")
async def get_validation(symbol: str):
    return validation_service.get(symbol.upper())