import base64
import io
import json
import re
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
import uvicorn
import joblib
import pandas as pd
import os
from api.models.LayoffFeatures import LayoffFeatures
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
from fastapi import Body


# ---------------------------------------------------
# Load the trained pipeline
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
pipeline_path = os.path.join(BASE_DIR, "trained_model", "layoff_pipeline.joblib")
print("load model pipeline", pipeline_path)
model_pipeline = joblib.load(pipeline_path)

# ---------------------------------------------------
# Load the Directories of Data and Training Metrics of the Model
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "layoffs.csv")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")

# ---------------------------------------------------
# Load the Directories of Data and Training Metrics of the Model
# ---------------------------------------------------
# Core paths (pathlib everywhere)
BASE_DIR = Path(__file__).resolve().parent  # .../api
DATA_DIR = BASE_DIR / "data"  # .../api/data
DATA_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLS = [
    "total_laid_off",
    "percentage_laid_off",
    "funds_raised",
    "industry",
    "country",
    "stage",
]


# ---------------------------------------------------
# Initialize FastAPI
# ---------------------------------------------------

app = FastAPI(
    title="Layoff Severity Prediction API",
    description="Predicts layoff severity class (low, medium, high, unknown).",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:8501"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------
#  FastAPI ENDPOINTS
# ---------------------------------------------------


@app.get("/")
def root():
    return {"message": "Layoff Severity Prediction API is running!"}


#
# Prediction endpoint


@app.post("/predict")
def predict_layoff(features: LayoffFeatures):
    # Create DataFrame from incoming data
    data = pd.DataFrame([features.dict()])

    # Predict using the model pipeline
    pred_class = model_pipeline.predict(data)[0]

    # Map numeric class to labels
    severity_map = {
        0: "Low (≤10%)",
        1: "Medium (10-50%)",
        2: "High (>50%)",
        3: "Unknown severity",
    }
    prediction_label = severity_map.get(pred_class, "Unknown")

    return {"predicted_class": int(pred_class), "severity_label": prediction_label}


@app.get("/data")
def get_data():
    df = pd.read_csv(DATA_PATH)
    df = df.replace([np.inf, -np.inf], None).where(pd.notnull(df), None)
    return Response(df.to_json(orient="records"), media_type="application/json")


#
# metrics endpoint
@app.get("/metrics")
def get_metrics(embed_images: bool = True):
    """Return metrics + confusion matrix (numbers + optional base64 PNG) in one payload."""
    metrics_path = os.path.join(METRICS_DIR, "metrics.json")
    if not os.path.isfile(metrics_path):
        raise HTTPException(
            status_code=404, detail="metrics.json not found. Re-run training."
        )

    with open(metrics_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Confusion matrix (optional, if present)
    cm_npy = os.path.join(METRICS_DIR, "confusion_matrix.npy")
    cm_png = os.path.join(METRICS_DIR, "confusion_matrix.png")

    cm_block = {}
    if os.path.isfile(cm_npy):
        cm_block["matrix"] = np.load(cm_npy).tolist()

    if embed_images and os.path.isfile(cm_png):
        with open(cm_png, "rb") as f:
            cm_block["png_b64"] = base64.b64encode(f.read()).decode(
                "ascii"
            )  # no data: prefix; smaller

    if cm_block:
        payload["confusion_matrix"] = cm_block

    return payload


### THE UPLOAD ENDPOINT
@app.post("/data/upload")
async def upload_csv(file: UploadFile = File(...)):
    # 1) basic validation
    fname = file.filename or "upload.csv"
    if not fname.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    #  read file body
    raw = await file.read()

    #  parse with pandas
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", fname)
    save_path = DATA_DIR / "uploads" / f"{ts}__{safe}"

    #  save EXACTLY what we parsed
    df.to_csv(save_path, index=False)

    return {
        "message": "CSV uploaded successfully",
        "saved_as": str(save_path.relative_to(BASE_DIR)),  # e.g. data/2025...__file.csv
        "rows": int(len(df)),
        "columns": list(df.columns),
    }


@app.post("/retrain")
def retrain():
    """
    Retrain the model. Optionally pass {"data_path": "data/uploads/<timestamp>__file.csv"}

    """


# ---------------------------------------------------
# Run with:
# uvicorn api.main:app --reload
# ---------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
