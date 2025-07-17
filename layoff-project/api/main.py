from fastapi import FastAPI
import uvicorn
import joblib
import pandas as pd
import os
from api.models.LayoffFeatures import LayoffFeatures

# ---------------------------------------------------
# Load the trained pipeline
# ---------------------------------------------------

pipeline_path = os.path.join(
    os.path.dirname(__file__), "..", "models", "layoff_pipeline.joblib"
)
pipeline_path = os.path.abspath(pipeline_path)

model_pipeline = joblib.load(pipeline_path)

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


# ---------------------------------------------------
# Run with:
# uvicorn api.main:app --reload
# ---------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
