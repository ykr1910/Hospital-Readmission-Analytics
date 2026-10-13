import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from io import StringIO
import psycopg2

# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# --------------------------------------------------
# 2. Create PostgreSQL connection
# --------------------------------------------------

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)

def load_dataframe_to_postgres(dataframe, table_name):
    csv_buffer = StringIO()
    dataframe.to_csv(csv_buffer, index=False, header=False)
    csv_buffer.seek(0)

    connection = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    cursor = connection.cursor()

    columns = ", ".join(dataframe.columns)

    cursor.copy_expert(
        f"COPY healthcare.{table_name} ({columns}) FROM STDIN WITH CSV",
        csv_buffer
    )

    connection.commit()
    cursor.close()
    connection.close()


# Clear existing data before loading fresh data
with engine.begin() as connection:
    connection.exec_driver_sql(
        "TRUNCATE TABLE healthcare.readmission_mart, "
        "healthcare.diagnoses, "
        "healthcare.admissions, "
        "healthcare.patient_summary"
    )

print("Existing database data cleared.")


# --------------------------------------------------
# 3. Load cleaned dataset
# --------------------------------------------------

file_path = "data/cleaned/cleaned_hospital_data.csv"

df = pd.read_csv(file_path)

print("Cleaned dataset loaded!")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# --------------------------------------------------
# 4. Prepare admissions table
# --------------------------------------------------

admissions_columns = [
    "encounter_id",
    "patient_key",
    "admission_type_id",
    "admission_source_id",
    "discharge_disposition_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "readmitted"
]

admissions = df[admissions_columns].copy()


# --------------------------------------------------
# 5. Prepare diagnoses table
# --------------------------------------------------

diagnoses = df[
    [
        "encounter_id",
        "diag_1",
        "diag_2",
        "diag_3",
        "diagnosis_count"
    ]
].copy()

diagnoses = diagnoses.rename(
    columns={
        "diag_1": "diagnosis_1",
        "diag_2": "diagnosis_2",
        "diag_3": "diagnosis_3"
    }
)


# --------------------------------------------------
# 6. Prepare patient summary table
# --------------------------------------------------

patient_summary = df[
    [
        "patient_key",
        "race",
        "gender",
        "age_group",
        "medical_specialty",
        "payer_code",
        "prior_visits"
    ]
].drop_duplicates(
    subset=["patient_key"]
).copy()


# --------------------------------------------------
# 7. Prepare readmission mart
# --------------------------------------------------

readmission_mart = df[
    [
        "encounter_id",
        "patient_key",
        "age_group",
        "gender",
        "admission_type_id",
        "time_in_hospital",
        "prior_visits",
        "diagnosis_count",
        "medication_count",
        "medical_specialty",
        "readmitted"
    ]
].copy()


# --------------------------------------------------
# 8. Load data into PostgreSQL
# --------------------------------------------------

print("\nLoading admissions...")
load_dataframe_to_postgres(admissions, "admissions")
print("Admissions loaded:", len(admissions))


print("\nLoading diagnoses...")
load_dataframe_to_postgres(diagnoses, "diagnoses")
print("Diagnoses loaded:", len(diagnoses))


print("\nLoading patient summary...")
load_dataframe_to_postgres(patient_summary, "patient_summary")
print("Patient summary loaded:", len(patient_summary))


print("\nLoading readmission mart...")
load_dataframe_to_postgres(readmission_mart, "readmission_mart")
print("Readmission mart loaded:", len(readmission_mart))


print("\nDatabase loading completed successfully!")