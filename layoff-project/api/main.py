import json
from fastapi import FastAPI, Response

from fastapi.encoders import jsonable_encoder
import uvicorn
import joblib
import pandas as pd
import os
from api.models.LayoffFeatures import LayoffFeatures
import numpy as np

# ---------------------------------------------------
# Load the trained pipeline
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
pipeline_path = os.path.join(BASE_DIR, "trained_model", "layoff_pipeline.joblib")
print("load model pipeline", pipeline_path)
model_pipeline = joblib.load(pipeline_path)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "layoffs.csv")
# ---------------------------------------------------
# Initialize FastAPI
# ---------------------------------------------------

app = FastAPI(
    title="Layoff Severity Prediction API",
    description="Predicts layoff severity class (low, medium, high, unknown).",
    version="1.0",
)


@app.get("/")
def root():
    return {"message": "Layoff Severity Prediction API is running!"}


# ---------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------


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


# ---------------------------------------------------
# Run with:
# uvicorn api.main:app --reload
# ---------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
