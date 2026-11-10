"""FastAPI inference service: prices one instrument per request."""
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.simulate import FEATURE_COLUMNS, YIELD_COLUMN
from src.models.train import load_latest_model, predict
from src.utils.helpers import ARTIFACT_DIR, read_metrics


class PriceRequest(BaseModel):
    features: List[float] = Field(..., min_length=len(FEATURE_COLUMNS), max_length=len(FEATURE_COLUMNS))
    treasury_10y_yield: float = Field(..., gt=0, lt=0.2)


def create_app(artifact_dir=ARTIFACT_DIR):
    app = FastAPI(title="Finance Pricing Engine")

    def load():
        try:
            return load_latest_model(artifact_dir)
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error))

    @app.get("/health")
    def health():
        try:
            _, _, name = load_latest_model(artifact_dir)
        except FileNotFoundError:
            name = None
        return {"status": "ok", "model": name}

    @app.post("/price")
    def price(request: PriceRequest):
        model, scaler, name = load()
        frame = pd.DataFrame([request.features + [request.treasury_10y_yield]],
                             columns=FEATURE_COLUMNS + [YIELD_COLUMN])
        return {"price": float(predict(model, scaler, frame)[0]), "model": name}

    @app.get("/metrics")
    def metrics():
        return read_metrics(["latest_drift_score", "pipeline_status", "model_rmse", "last_retrain"])

    return app


app = create_app()
