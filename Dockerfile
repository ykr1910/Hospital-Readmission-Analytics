
FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ ./api/
COPY models/registry_model_v1.joblib ./models/best_model.joblib
COPY models/decision_threshold.json ./models/decision_threshold.json
COPY models/reference_features.csv ./models/reference_features.csv
COPY models/deployment_manifest.json ./models/deployment_manifest.json

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
