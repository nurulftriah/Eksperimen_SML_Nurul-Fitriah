import os
import pickle
import psutil
import time
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(
    title="Credit Scoring API",
    description="Served for Dicoding SML project monitoring"
)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully from model.pkl")
else:
    model = None
    print("WARNING: model.pkl not found! Predictions will fail until model.pkl is provided.")

# Prometheus Metrics
REQUEST_COUNTER = Counter("http_requests_total", "Total number of HTTP requests to predict endpoint")
CPU_GAUGE = Gauge("system_cpu_usage", "System CPU usage percentage")
RAM_GAUGE = Gauge("system_ram_usage", "System RAM usage percentage")
LATENCY_SUMMARY = Summary("model_latency_seconds", "Inference latency in seconds")


class CreditScoringInput(BaseModel):
    age: float
    annual_income: float
    credit_score: float
    loan_amount: float
    marital_status_Married: int
    marital_status_Single: int
    education_level_High_School: int
    education_level_Master: int
    education_level_PhD: int
    age_group_Middle_Aged: int
    age_group_Senior: int


@app.post("/predict")
def predict(data: CreditScoringInput):
    REQUEST_COUNTER.inc()
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")

    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Remap nama kolom agar cocok dengan kolom hasil pd.get_dummies saat training
    column_mapping = {
        "education_level_High_School": "education_level_High School",
        "age_group_Middle_Aged": "age_group_Middle-Aged"
    }
    input_df = input_df.rename(columns=column_mapping)

    try:
        start_time = time.time()
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        duration = time.time() - start_time
        LATENCY_SUMMARY.observe(duration)

        return {
            "prediction": prediction,
            "probability_default": probability,
            "label": "Gagal Bayar" if prediction == 1 else "Lancar"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.get("/metrics")
def metrics():
    CPU_GAUGE.set(psutil.cpu_percent(interval=None))
    RAM_GAUGE.set(psutil.virtual_memory().percent)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}
