from fastapi import APIRouter, HTTPException
from app.services.data_service import data_service
from app.services.indicator_service import indicator_service

router = APIRouter()

@router.get("/{symbol}/data")
async def get_stock_data(symbol: str):
    try:
        df = await data_service.get_ohlcv(symbol.upper())
        return {
            "symbol": symbol.upper(),
            "candles": df[["Date","Open","High","Low","Close","Volume"]]
                       .to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(400, str(e))

@router.get("/{symbol}/quote")
async def get_quote(symbol: str):
    return await data_service.get_latest_price(symbol.upper())

@router.get("/{symbol}/indicators")
async def get_indicators(symbol: str):
    return await indicator_service.compute_all(symbol.upper())