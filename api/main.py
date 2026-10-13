
import csv
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "decision_threshold.json"
MANIFEST_PATH = PROJECT_ROOT / "models" / "deployment_manifest.json"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "prediction_logs.csv"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("readmission_api")

FEATURE_FIELDS = [
    "age_group",
    "gender",
    "admission_type_id",
    "time_in_hospital",
    "prior_visits",
    "diagnosis_count",
    "medication_count",
    "medical_specialty",
]

LOG_FIELDS = [
    "timestamp_utc",
    *FEATURE_FIELDS,
    "probability",
    "threshold",
    "predicted_class",
    "latency_ms",
    "status",
    "error",
]


def log_prediction(record: dict) -> None:
    """Append one prediction or error event to the CSV log."""
    try:
        new_file = not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0

        with LOG_FILE.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=LOG_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(record)

    except OSError:
        logger.exception("Could not write prediction monitoring log")



import os


MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local").lower()

if MODEL_SOURCE == "registry":
    import mlflow
    import mlflow.sklearn
    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "sqlite:///E:/Hospital_Readmission_Analytics/mlflow.db",
        )
    )

    REGISTERED_MODEL_URI = os.getenv(
        "MLFLOW_MODEL_URI",
        "models:/HospitalReadmissionXGBoost/1",
    )

    model = mlflow.sklearn.load_model(REGISTERED_MODEL_URI)
    logger.info("Loaded model from MLflow Registry: %s", REGISTERED_MODEL_URI)

elif MODEL_SOURCE == "local":
    model = joblib.load(MODEL_PATH)
    logger.info("Loaded model from local artifact: %s", MODEL_PATH)

else:
    raise ValueError(
        "MODEL_SOURCE must be either 'local' or 'registry'"
    )


with THRESHOLD_PATH.open("r", encoding="utf-8") as file:
    model_config = json.load(file)

with MANIFEST_PATH.open("r", encoding="utf-8-sig") as file:
    deployment_manifest = json.load(file)

THRESHOLD = float(model_config["threshold"])

app = FastAPI(
    title="Hospital Readmission Prediction API",
    description=(
        "Academic prototype for predicting 30-day hospital "
        "readmission. Not for clinical decision-making."
    ),
    version="1.1.0",
)


class EncounterInput(BaseModel):
    age_group: str = Field(description="Age category, e.g. 70-80")
    gender: str = Field(description="Gender category")
    admission_type_id: int = Field(ge=1, le=8)
    time_in_hospital: int = Field(ge=1, le=14)
    prior_visits: int = Field(ge=0)
    diagnosis_count: int = Field(ge=1, le=16)
    medication_count: int = Field(ge=0)
    medical_specialty: str = Field(description="Medical specialty category")


@app.get("/")
def home():
    return {
        "message": "Hospital Readmission Prediction API",
        "status": "running",
        "purpose": "Academic decision-support prototype only",
    }



@app.get("/health")
def health():
    registry_info = deployment_manifest.get("model_registry", {})

    return {
        "status": "healthy",
        "model": model_config["model_name"],
        "model_version": deployment_manifest["model_version"],
        "model_registry": registry_info,
        "deployment_stage": deployment_manifest["deployment_stage"],
        "threshold": THRESHOLD,
        "evaluation_metrics": deployment_manifest["metrics"],
    }


@app.get("/metrics")
def metrics():
    if not LOG_FILE.exists():
        rows = []
    else:
        with LOG_FILE.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

    total = len(rows)
    successful = sum(
        row.get("status") == "success" for row in rows
    )
    errors = total - successful

    latencies = [
        float(row["latency_ms"])
        for row in rows
        if row.get("latency_ms")
    ]

    return {
        "total_requests": total,
        "successful_predictions": successful,
        "failed_requests": errors,
        "error_rate_percent": round(
            errors / total * 100, 2
        ) if total else 0.0,
        "average_latency_ms": round(
            sum(latencies) / len(latencies), 3
        ) if latencies else 0.0,
        "latest_prediction_timestamp_utc": (
            rows[-1].get("timestamp_utc") if rows else None
        ),
    }



REFERENCE_FILE = PROJECT_ROOT / "models" / "reference_features.csv"

NUMERIC_FEATURES = [
    "time_in_hospital",
    "prior_visits",
    "diagnosis_count",
    "medication_count",
]

CATEGORICAL_FEATURES = [
    "age_group",
    "gender",
    "admission_type_id",
    "medical_specialty",
]


def calculate_psi(reference, current, bins=10):
    """Calculate PSI using reference-data quantile bins."""
    reference = pd.to_numeric(
        pd.Series(reference), errors="coerce"
    ).dropna().to_numpy(dtype=float)

    current = pd.to_numeric(
        pd.Series(current), errors="coerce"
    ).dropna().to_numpy(dtype=float)

    if len(reference) == 0 or len(current) == 0:
        return None

    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current, bins=edges)[0]

    ref_pct = np.clip(ref_counts / len(reference), 1e-6, None)
    cur_pct = np.clip(cur_counts / len(current), 1e-6, None)

    return float(
        np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    )


def calculate_categorical_drift(reference, current):
    """Calculate total variation distance between categories."""
    ref = pd.Series(reference).fillna("Missing").astype(str)
    cur = pd.Series(current).fillna("Missing").astype(str)

    if ref.empty or cur.empty:
        return None

    categories = set(ref.unique()) | set(cur.unique())

    ref_dist = ref.value_counts(normalize=True)
    cur_dist = cur.value_counts(normalize=True)

    distance = sum(
        abs(ref_dist.get(category, 0) - cur_dist.get(category, 0))
        for category in categories
    ) / 2

    return float(distance)



@app.get("/drift")
def drift(
    window: int = Query(default=1000, ge=30, le=10000)
):
    if not REFERENCE_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail="Reference feature dataset is missing.",
        )

    reference = pd.read_csv(REFERENCE_FILE)

    if not LOG_FILE.exists():
        current = pd.DataFrame()
    else:
        current = pd.read_csv(LOG_FILE)

    # Keep only successful predictions.
    if not current.empty and "status" in current.columns:
        current = current[current["status"] == "success"].copy()

    # Keep only the most recent successful predictions.
    current = current.tail(window)

    minimum_samples = 30

    # Do not calculate or display drift scores with too little data.
    if len(current) < minimum_samples:
        return {
            "status": "insufficient_data",
            "current_samples": len(current),
            "minimum_samples": minimum_samples,
            "window": window,
            "reference_samples": len(reference),
            "message": (
                "Collect at least 30 successful predictions "
                "before interpreting drift results."
            ),
            "features": {},
        }

    feature_results = {}

    for feature in NUMERIC_FEATURES:
        if feature not in reference.columns or feature not in current.columns:
            continue

        value = calculate_psi(
            reference[feature],
            current[feature],
        )

        feature_results[feature] = {
            "metric": "PSI",
            "value": round(value, 4) if value is not None else None,
            "status": (
                "review"
                if value is not None and value >= 0.2
                else "ok"
            ),
        }

    for feature in CATEGORICAL_FEATURES:
        if feature not in reference.columns or feature not in current.columns:
            continue

        value = calculate_categorical_drift(
            reference[feature],
            current[feature],
        )

        feature_results[feature] = {
            "metric": "total_variation_distance",
            "value": round(value, 4) if value is not None else None,
            "status": (
                "review"
                if value is not None and value >= 0.2
                else "ok"
            ),
        }

    has_drift = any(
        item["status"] == "review"
        for item in feature_results.values()
    )

    return {
        "status": "review" if has_drift else "ok",
        "current_samples": len(current),
        "minimum_samples": minimum_samples,
        "window": window,
        "reference_samples": len(reference),
        "features": feature_results,
        "notice": (
            "Drift indicates distribution changes, not model accuracy "
            "or clinical performance."
        ),
    }




@app.post("/predict")
def predict(encounter: EncounterInput):
    start_time = time.perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat()

    probability = None
    prediction = None
    status = "success"
    error_message = ""

    try:
        input_data = pd.DataFrame([encounter.model_dump()])
        probability = float(model.predict_proba(input_data)[0, 1])
        prediction = int(probability >= THRESHOLD)

        return {
            "predicted_readmission_within_30_days": prediction,
            "probability": round(probability, 4),
            "threshold": round(THRESHOLD, 4),
            "model": model_config["model_name"],
            "notice": (
                "Academic prototype only; this output is not "
                "medical advice or a validated clinical assessment."
            ),
        }

    except Exception as exc:
        status = "error"
        error_message = str(exc)
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check API logs for details.",
        ) from exc



    finally:
        latency_ms = round(
            (time.perf_counter() - start_time) * 1000, 3
        )

        input_features = encounter.model_dump()

        log_record = {
            "timestamp_utc": timestamp,
            **input_features,
            "probability": probability,
            "threshold": THRESHOLD,
            "predicted_class": prediction,
            "latency_ms": latency_ms,
            "status": status,
            "error": error_message,
        }

        log_prediction(log_record)

        logger.info(
            "prediction_status=%s latency_ms=%.3f",
            status,
            latency_ms,
        )
