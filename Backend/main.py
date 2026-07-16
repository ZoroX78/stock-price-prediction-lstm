from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from api.v1 import stocks, predictions, watchlist

app = FastAPI(title=settings.APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

prefix = settings.API_V1_PREFIX
app.include_router(stocks.router,       prefix=f"{prefix}/stocks",      tags=["stocks"])
app.include_router(predictions.router,  prefix=f"{prefix}/predict",     tags=["predictions"])
app.include_router(watchlist.router,    prefix=f"{prefix}/watchlist",   tags=["watchlist"])

@app.on_event("startup")
async def startup():
    from app.core.cache import redis_cache
    await redis_cache.connect()

@app.get("/")
async def root():
    return {"status": "ok", "service": settings.APP_NAME}