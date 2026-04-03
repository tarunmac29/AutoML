"""
api.py
FastAPI REST API to serve the trained AutoML model.

Endpoints:
  GET  /         → health check
  GET  /info     → model metadata
  POST /predict  → single prediction
  POST /predict/batch → batch predictions (CSV upload)

Run with:
  uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import json
import io

import numpy as np
import pandas as pd
import joblib
import uvicorn

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Smart AutoML API",
    description="REST API for serving trained AutoML models.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loading ─────────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.pkl")
META_PATH = os.environ.get("META_PATH", "models/model_meta.json")

_model = None
_meta = {}


def load_model():
    """Load model and metadata at startup."""
    global _model, _meta
    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model not found at {MODEL_PATH}. Train a model first via the Streamlit app.")
        return

    _model = joblib.load(MODEL_PATH)
    print(f"[INFO] Model loaded from {MODEL_PATH}: {type(_model).__name__}")

    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            _meta = json.load(f)
        print(f"[INFO] Metadata loaded: {_meta}")


@app.on_event("startup")
def startup_event():
    load_model()


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        example={"feature1": 1.5, "feature2": 3.2, "feature3": 0},
        description="Key-value pairs of feature name → value"
    )

class PredictResponse(BaseModel):
    prediction: Any
    prediction_label: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    model_name: Optional[str] = None
    problem_type: Optional[str] = None


class BatchPredictResponse(BaseModel):
    predictions: List[Any]
    count: int


class InfoResponse(BaseModel):
    model_name: str
    problem_type: str
    feature_names: List[str]
    status: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_name": type(_model).__name__ if _model else None,
    }


@app.get("/info", response_model=InfoResponse, tags=["Model Info"])
def model_info():
    """Return model metadata."""
    if _model is None:
        raise HTTPException(status_code=503, detail="No model loaded. Train a model first.")

    return {
        "model_name": type(_model).__name__,
        "problem_type": _meta.get("problem_type", "unknown"),
        "feature_names": _meta.get("feature_names", []),
        "status": "ready",
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Make a single prediction.

    Body: JSON with 'features' dict mapping feature names to values.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="No model loaded. Train a model first.")

    try:
        input_df = pd.DataFrame([request.features])

        # Reorder columns to match training if metadata available
        feature_names = _meta.get("feature_names", [])
        if feature_names:
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_names]

        prediction = _model.predict(input_df)[0]
        prediction_python = prediction.item() if hasattr(prediction, "item") else prediction

        response = {
            "prediction": prediction_python,
            "model_name": type(_model).__name__,
            "problem_type": _meta.get("problem_type", "unknown"),
        }

        # Decode label if encoder info available
        label_map = _meta.get("label_map", {})
        if label_map:
            response["prediction_label"] = label_map.get(str(int(prediction_python)), str(prediction_python))

        # Probabilities (classification)
        if hasattr(_model, "predict_proba"):
            proba = _model.predict_proba(input_df)[0]
            classes = _meta.get("classes", [str(i) for i in range(len(proba))])
            response["probabilities"] = {str(cls): round(float(p), 4) for cls, p in zip(classes, proba)}

        return response

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(file: UploadFile = File(...)):
    """
    Batch prediction from uploaded CSV file.

    Upload a CSV with feature columns (no target column needed).
    Returns list of predictions.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="No model loaded.")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted for batch prediction.")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        feature_names = _meta.get("feature_names", [])
        if feature_names:
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[feature_names]

        predictions = _model.predict(df)
        preds_list = [p.item() if hasattr(p, "item") else p for p in predictions]

        return {"predictions": preds_list, "count": len(preds_list)}

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Batch prediction failed: {str(e)}")


# ── Save metadata helper (called from webapp after training) ──────────────────

def save_model_metadata(
    model,
    feature_names: list,
    problem_type: str,
    label_encoder=None,
    path: str = "models/model_meta.json"
):
    """
    Save model metadata JSON alongside the .pkl file.
    Used by webapp.py after training to enable the API.

    Args:
        model: Trained model
        feature_names: List of feature column names
        problem_type: 'classification' or 'regression'
        label_encoder: Optional sklearn LabelEncoder
        path: Where to save the JSON
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    meta = {
        "model_name": type(model).__name__,
        "problem_type": problem_type,
        "feature_names": feature_names,
    }

    if label_encoder is not None:
        classes = label_encoder.classes_.tolist()
        meta["classes"] = [str(c) for c in classes]
        meta["label_map"] = {str(i): str(c) for i, c in enumerate(classes)}

    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
