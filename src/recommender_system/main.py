from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel

from recommender_system.config import RAW_DIR
from recommender_system.data.data_loader import DataLoader, DataPaths
from recommender_system.recommender import Recommender, RecommendationResult


# ---------- Response Models ----------

class HealthResponse(BaseModel):
    status: str


class RecommendationItem(BaseModel):
    item_id: str
    title: str = ""
    score: float


class PopularResponse(BaseModel):
    k: int
    items: List[RecommendationItem]


class RecommendationsResponse(BaseModel):
    user_id: str
    k: int
    fallback_used: bool
    items: List[RecommendationItem]


# ---------- App Lifespan (load once on startup) ----------

def _get_raw_dir_override():
    # Optional override, e.g. export DATA_RAW_DIR="/path/to/data/raw"
    override = os.getenv("DATA_RAW_DIR")
    return override


@asynccontextmanager
async def lifespan(app: FastAPI):
    raw_dir = _get_raw_dir_override() or str(RAW_DIR)
    paths = DataPaths.from_raw_dir(raw_dir=__import__("pathlib").Path(raw_dir))

    loader = DataLoader(paths)
    users_df, items_df, events_df = loader.load_all()

    app.state.recommender = Recommender(users_df, items_df, events_df)
    yield
    # no teardown needed


app = FastAPI(title="recommender_system", version="0.1.0", lifespan=lifespan)


def _get_recommender(app: FastAPI) -> Recommender:
    rec = getattr(app.state, "recommender", None)
    if rec is None:
        raise HTTPException(status_code=503, detail="Recommender not ready")
    return rec


# ---------- Endpoints ----------

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.get("/popular", response_model=PopularResponse)
def popular(
    k: int = Query(10, ge=1, le=100, description="Number of items to return")
):
    rec = _get_recommender(app)
    items = rec.recommend_popular(k=k)
    return {"k": k, "items": items}


@app.get("/recommendations", response_model=RecommendationsResponse)
def recommendations(
    user_id: str = Query(..., min_length=1, description="User ID"),
    k: int = Query(10, ge=1, le=100, description="Number of items to return"),
):
    rec = _get_recommender(app)
    result: RecommendationResult = rec.recommend_for_user(user_id=user_id, k=k)
    return {
        "user_id": user_id,
        "k": k,
        "fallback_used": result.fallback_used,
        "items": result.items,
    }