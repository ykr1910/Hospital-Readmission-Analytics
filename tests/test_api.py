
import csv

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api


client = TestClient(api.app)


@pytest.fixture
def isolated_files(tmp_path, monkeypatch):
    """Keep test logs and reference data separate from project data."""
    log_file = tmp_path / "prediction_logs.csv"
    reference_file = tmp_path / "reference_features.csv"

    monkeypatch.setattr(api, "LOG_FILE", log_file)
    monkeypatch.setattr(api, "REFERENCE_FILE", reference_file)

    reference = pd.DataFrame({
        "age_group": ["50-60", "60-70", "70-80", "40-50"] * 25,
        "gender": ["Male", "Female"] * 50,
        "admission_type_id": [1, 2, 3, 4] * 25,
        "time_in_hospital": [1, 3, 5, 7] * 25,
        "prior_visits": [0, 1, 2, 3] * 25,
        "diagnosis_count": [1, 2, 3, 4] * 25,
        "medication_count": [0, 5, 10, 15] * 25,
        "medical_specialty": [
            "InternalMedicine",
            "Cardiology",
            "Surgery",
            "Unknown",
        ] * 25,
    })
    reference.to_csv(reference_file, index=False)

    yield log_file, reference_file


def make_prediction_rows(count=30):
    """Build successful prediction-log rows for drift tests."""
    rows = []

    for i in range(count):
        rows.append({
            "age_group": ["50-60", "60-70"][i % 2],
            "gender": ["Male", "Female"][i % 2],
            "admission_type_id": (i % 4) + 1,
            "time_in_hospital": (i % 7) + 1,
            "prior_visits": i % 4,
            "diagnosis_count": (i % 4) + 1,
            "medication_count": (i % 4) * 5,
            "medical_specialty": [
                "InternalMedicine",
                "Cardiology",
                "Surgery",
                "Unknown",
            ][i % 4],
            "status": "success",
            "latency_ms": 20.0,
            "timestamp_utc": "2026-10-15T00:00:00+00:00",
        })

    return rows


def write_log(path, rows):
    """Write test rows to a CSV file."""
    if not rows:
        pd.DataFrame(columns=api.LOG_FIELDS).to_csv(
            path,
            index=False,
        )
        return

    pd.DataFrame(rows).to_csv(path, index=False)


def test_health_returns_model_metadata():
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "healthy"
    assert body["model"]
    assert body["model_version"]
    assert 0 <= body["threshold"] <= 1
    assert "evaluation_metrics" in body


def test_predict_returns_prediction_and_logs_success(
    isolated_files,
    monkeypatch,
):
    log_file, _ = isolated_files

    class FakeModel:
        def predict_proba(self, frame):
            assert len(frame) == 1
            return np.array([[0.70, 0.30]])

    monkeypatch.setattr(api, "model", FakeModel())

    payload = {
        "age_group": "50-60",
        "gender": "Male",
        "admission_type_id": 1,
        "time_in_hospital": 5,
        "prior_visits": 2,
        "diagnosis_count": 4,
        "medication_count": 10,
        "medical_specialty": "InternalMedicine",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    body = response.json()
    assert body["predicted_readmission_within_30_days"] == 1
    assert body["probability"] == 0.30
    assert "notice" in body

    with log_file.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["status"] == "success"
    assert rows[0]["age_group"] == "50-60"


def test_predict_rejects_invalid_input(isolated_files):
    payload = {
        "age_group": "50-60",
        "gender": "Male",
        "admission_type_id": 99,
        "time_in_hospital": 5,
        "prior_visits": 2,
        "diagnosis_count": 4,
        "medication_count": 10,
        "medical_specialty": "InternalMedicine",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_metrics_counts_successes_and_failures(isolated_files):
    log_file, _ = isolated_files

    rows = [
        {
            "status": "success",
            "latency_ms": 20.0,
            "timestamp_utc": "2026-10-15T00:00:00+00:00",
        },
        {
            "status": "error",
            "latency_ms": 40.0,
            "timestamp_utc": "2026-10-15T00:01:00+00:00",
        },
    ]

    write_log(log_file, rows)

    response = client.get("/metrics")

    assert response.status_code == 200

    body = response.json()
    assert body["total_requests"] == 2
    assert body["successful_predictions"] == 1
    assert body["failed_requests"] == 1
    assert body["error_rate_percent"] == 50.0
    assert body["average_latency_ms"] == 30.0


def test_drift_requires_minimum_samples(isolated_files):
    log_file, _ = isolated_files
    write_log(log_file, make_prediction_rows(29))

    response = client.get("/drift")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "insufficient_data"
    assert body["current_samples"] == 29
    assert body["minimum_samples"] == 30
    assert body["features"] == {}


def test_drift_calculates_all_features_at_minimum(
    isolated_files,
):
    log_file, _ = isolated_files
    write_log(log_file, make_prediction_rows(30))

    response = client.get("/drift")

    assert response.status_code == 200

    body = response.json()
    assert body["current_samples"] == 30
    assert body["status"] in {"ok", "review"}

    expected_features = set(api.EXPECTED_FEATURES)
    assert set(body["features"]) == expected_features

    for feature in api.NUMERIC_FEATURES:
        result = body["features"][feature]
        assert result["metric"] == "PSI"
        assert result["status"] in {"ok", "review"}

    for feature in api.CATEGORICAL_FEATURES:
        result = body["features"][feature]
        assert result["metric"] == "total_variation_distance"
        assert result["status"] in {"ok", "review"}


def test_drift_rejects_missing_reference_feature(
    isolated_files,
):
    _, reference_file = isolated_files

    reference = pd.read_csv(reference_file)
    reference = reference.drop(columns=["admission_type_id"])
    reference.to_csv(reference_file, index=False)

    response = client.get("/drift")

    assert response.status_code == 500

    detail = response.json()["detail"]
    assert "admission_type_id" in detail["missing_features"]


def test_drift_rejects_missing_current_feature(
    isolated_files,
):
    log_file, _ = isolated_files

    rows = make_prediction_rows(30)
    rows = [
        {
            key: value
            for key, value in row.items()
            if key != "admission_type_id"
        }
        for row in rows
    ]
    write_log(log_file, rows)

    response = client.get("/drift")

    assert response.status_code == 500

    detail = response.json()["detail"]
    assert "admission_type_id" in detail["missing_features"]


def test_drift_rejects_invalid_window():
    response = client.get("/drift?window=10")

    assert response.status_code == 422
