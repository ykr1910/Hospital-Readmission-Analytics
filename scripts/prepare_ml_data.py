
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

url = (
    f"postgresql+psycopg://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}@127.0.0.1:"
    f"{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

engine = create_engine(url)

query = """
SELECT
    encounter_id,
    patient_key,
    age_group,
    gender,
    admission_type_id,
    time_in_hospital,
    prior_visits,
    diagnosis_count,
    medication_count,
    readmitted,
    medical_specialty
FROM healthcare.readmission_mart
"""

df = pd.read_sql(query, engine)
engine.dispose()

print("Data loaded:", df.shape)

# Keep only encounters with known readmission labels
df = df[df["readmitted"].isin(["<30", ">30", "NO"])].copy()

# Binary target: readmitted within 30 days
df["target"] = (df["readmitted"] == "<30").astype(int)

# Save patient identifier separately for group-aware splitting
patient_groups = df["patient_key"].copy()

# Exclude labels and identifiers from model features
X = df.drop(
    columns=["encounter_id", "patient_key", "readmitted", "target"]
)
y = df["target"]

# Save reproducible intermediate datasets
os.makedirs("data/ml", exist_ok=True)

X.to_csv("data/ml/features.csv", index=False)
y.to_frame().to_csv("data/ml/target.csv", index=False)
patient_groups.to_frame().to_csv(
    "data/ml/patient_groups.csv", index=False
)

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts().sort_index())

print("\nPositive-class rate:", round(y.mean() * 100, 2), "%")
print("\nMissing values in features:")
print(X.isnull().sum())

print("\nSaved files under data/ml/")
