import json
import os
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Breast Cancer Predictor")

FEATURES = [
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
]


class PatientInput(BaseModel):
    # ranges are just sanity limits, wrong types / missing fields are rejected by pydantic
    mean_radius: float = Field(gt=0, lt=50)
    mean_texture: float = Field(gt=0, lt=100)
    mean_perimeter: float = Field(gt=0, lt=300)
    mean_area: float = Field(gt=0, lt=5000)
    mean_smoothness: float = Field(gt=0, lt=1)


def artifact_dir() -> Path:
    # read each time so tests can point it somewhere else
    return Path(os.getenv("MODEL_DIR", "artifacts"))


def load_artifacts():
    """Load model + metadata. Raises 503 if missing or not compatible."""
    d = artifact_dir()
    model_path = d / "model.joblib"
    meta_path = d / "metadata.json"

    if not model_path.exists() or not meta_path.exists():
        raise HTTPException(503, "Model artifacts not found. Run train.py first.")

    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        raise HTTPException(503, "metadata.json is corrupted.")

    # compatibility checks
    if meta.get("features") != FEATURES:
        raise HTTPException(503, "Model features do not match the API features.")
    if meta.get("sklearn_version") != sklearn.__version__:
        raise HTTPException(
            503,
            f"Model was trained with scikit-learn {meta.get('sklearn_version')} "
            f"but {sklearn.__version__} is installed. Retrain the model.",
        )

    try:
        model = joblib.load(model_path)
    except Exception:
        raise HTTPException(503, "Model file could not be loaded.")
    return model, meta


@app.get("/")
def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    _, meta = load_artifacts()
    return meta


@app.post("/predict")
def predict(data: PatientInput):
    model, meta = load_artifacts()
    row = pd.DataFrame([data.model_dump()])[FEATURES]
    pred = int(model.predict(row)[0])
    result = {"prediction": pred, "label": meta["classes"][str(pred)]}
    if hasattr(model, "predict_proba"):
        result["probability"] = round(float(model.predict_proba(row)[0][pred]), 4)
    return result
