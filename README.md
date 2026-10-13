Hospital Readmission Analytics and Prediction
Phase I – Data Engineering and MLOps

Student: Yash Kumar Rajput
USN: 1CR23CI063
Institution: CMR Institute of Technology, Bengaluru
Course: ML Ops
Semester: 7th Semester

1. Project Overview

This project implements a Phase I data engineering pipeline for analysing hospital readmissions using a public anonymized hospital encounter dataset.

The pipeline acquires the raw dataset, creates a staging copy, performs data-quality validation and transformation, engineers analytical features, loads curated data into PostgreSQL, and presents analytical results through a Streamlit dashboard.

Apache Airflow is used in WSL Ubuntu to orchestrate the repeatable ETL workflow.

Phase I scope: data acquisition, ingestion, transformation, validation, PostgreSQL analytics, orchestration, and dashboarding.

Machine-learning model training, MLflow, FastAPI, Docker deployment, and monitoring belong to Phase II and are intentionally not included in this Phase I implementation.

2. Dataset

Dataset: Diabetes 130-US Hospitals for Years 1999-2008

Source: UCI Machine Learning Repository
Dataset ID: 296
DOI: 10.24432/C5230J
License: CC BY 4.0

The dataset contains 101,766 hospital encounters from 130 US hospitals/integrated delivery networks over ten years. The data is de-identified and contains patient and encounter attributes relevant to readmission analysis.

The project uses the dataset's readmitted field to identify early readmission:

<30 = readmitted within 30 days
>30 = readmitted after 30 days
NO = no readmission
3. Architecture
Public Anonymized Dataset
          |
          v
      Raw Layer
          |
          v
     ingest.py
          |
          v
    Staging Layer
          |
          v
   transform.py
          |
          +------> Validation / Rejected Records Log
          |
          v
     Cleaned Layer
          |
          v
 load_database.py
          |
          v
 PostgreSQL (healthcare schema)
          |
          +------> Analytical Tables / Readmission Mart
          |
          v
 Streamlit Dashboard

Airflow orchestrates:
ingest_data -> transform_and_validate -> load_postgresql

The pipeline follows a layered data-engineering architecture:

Raw Layer – preserves the original downloaded dataset.
Staging Layer – stores the ingestion output and metadata.
Cleaned Layer – contains validated and transformed data.
PostgreSQL Layer – stores curated analytical tables.
Dashboard Layer – provides interactive analytics through Streamlit.
Airflow – orchestrates the complete ETL workflow.
4. Project Structure
Hospital_Readmission_Analytics/
│
├── data/
│   ├── raw/
│   │   └── diabetic_data.csv
│   │
│   ├── staging/
│   │   ├── diabetic_data_staging.csv
│   │   └── ingestion_metadata.csv
│   │
│   └── cleaned/
│       └── cleaned_hospital_data.csv
│
├── logs/
│   └── rejected_records.csv
│
├── scripts/
│   ├── ingest.py
│   ├── transform.py
│   ├── load_database.py
│   ├── inspect_data.py
│   └── test_db.py
│
├── dashboard/
│   └── app.py
│
├── README.md
├── requirements.txt
└── .env

Apache Airflow is installed separately in WSL Ubuntu and the DAG is stored in the Airflow DAG directory:

~/airflow/dags/hospital_readmission_etl.py
5. Technologies Used
Technology	Purpose
Python	ETL and data processing
Pandas	Data cleaning and transformation
PostgreSQL	Relational analytical database
SQLAlchemy	Database connectivity
Psycopg2	PostgreSQL database operations
Apache Airflow	ETL orchestration
Streamlit	Interactive dashboard
Plotly	Data visualization
python-dotenv	Environment variable management
WSL Ubuntu	Airflow execution environment
6. Environment Setup

The project uses separate Python environments for the Windows-based ETL/dashboard workflow and Apache Airflow.

Windows Python Environment

The main project uses a Python virtual environment:

venv/

Activate it using:

.\venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
Airflow Environment

Apache Airflow is executed through WSL Ubuntu because Airflow is not natively supported on Windows.

The Airflow virtual environment is:

/mnt/e/Hospital_Readmission_Analytics/airflow-venv

Activate it in WSL:

source /mnt/e/Hospital_Readmission_Analytics/airflow-venv/bin/activate
7. Database Configuration

Database connection details are stored in a .env file.

Example:

DB_HOST=localhost
DB_PORT=5432
DB_NAME=hospital_readmission
DB_USER=postgres
DB_PASSWORD=YOUR_PASSWORD

The .env file should not be committed to version control because it contains database credentials.

The PostgreSQL database used by the project is:

Database: hospital_readmission
Schema: healthcare
8. ETL Pipeline

The ETL process consists of three main scripts.

8.1 Data Ingestion

Run:

python scripts/ingest.py

The ingestion script:

Reads the raw dataset.
Preserves the raw dataset.
Creates a staging copy.
Records ingestion metadata.
Records extraction date.
Records source file.
Records status.
Records row and column counts.

Example successful output:

Raw dataset loaded successfully!
Rows: 101766
Columns: 50

Staging dataset created successfully!
Staging file: data/staging/diabetic_data_staging.csv

Ingestion metadata:
       extraction_date                 source_file   status  row_count  column_count
0  2026-09-03 19:45:03  data/raw/diabetic_data.csv  SUCCESS     101766           50
8.2 Transformation and Validation

Run:

python scripts/transform.py

The transformation stage performs:

Required-field validation.
Duplicate encounter validation.
Length-of-stay validation.
Diagnosis-count validation.
Readmission-field validation.
Missing-value handling.
Removal of highly incomplete weight field.
Standardization of selected categorical missing values.
Creation of patient-safe surrogate keys.
Prior-visit feature engineering.
Diagnosis-count feature engineering.
Medication-count feature engineering.
Age-group feature engineering.
Rejected-record logging.

The generated rejected-record file is:

logs/rejected_records.csv

Successful execution produced:

Rows after transformation: 101766
Columns after transformation: 54

New columns:
['patient_key', 'prior_visits', 'diagnosis_count', 'medication_count', 'age_group']

Cleaned dataset saved successfully!
Output file: data/cleaned/cleaned_hospital_data.csv
8.3 PostgreSQL Loading

Run:

python scripts/load_database.py

The loading stage:

Clears previous analytical data.
Loads the cleaned dataset.
Populates PostgreSQL analytical tables.
Uses bulk loading for efficient database insertion.
Creates the readmission analytical mart.

Successful execution:

Existing database data cleared.
Cleaned dataset loaded!
Rows: 101766
Columns: 54

Loading admissions...
Admissions loaded: 101766

Loading diagnoses...
Diagnoses loaded: 101766

Loading patient summary...
Patient summary loaded: 71518

Loading readmission mart...
Readmission mart loaded: 101766

Database loading completed successfully!
9. PostgreSQL Analytical Tables

The PostgreSQL database contains four analytical tables.

9.1 healthcare.admissions

Stores encounter-level admission and hospital utilization information.

Rows: 101,766

Important fields include:

admission_type_id
admission_source_id
discharge_disposition_id
time_in_hospital
num_lab_procedures
num_procedures
num_medications
number_outpatient
number_emergency
number_inpatient
readmitted
9.2 healthcare.diagnoses

Stores diagnosis information for each encounter.

Rows: 101,766

Important fields:

encounter_id
diagnosis_1
diagnosis_2
diagnosis_3
diagnosis_count
9.3 healthcare.patient_summary

Stores patient-level demographic and encounter-history information.

Rows: 71,518

Important fields:

patient_key
race
gender
age_group
medical_specialty
payer_code
prior_visits
9.4 healthcare.readmission_mart

The main analytical data mart used by the dashboard.

Rows: 101,766

Fields:

encounter_id
patient_key
age_group
gender
admission_type_id
time_in_hospital
prior_visits
diagnosis_count
medication_count
readmitted
medical_specialty
10. Data Quality and Validation

The transformation pipeline performs validation before creating the cleaned dataset.

Validation Rules
Validation	Rule
Encounter ID	Must not be missing
Duplicate encounter	Duplicate encounter IDs are rejected
Length of stay	Must be between 1 and 14 days
Number of diagnoses	Must be between 1 and 16
Readmission status	Must not be missing

Invalid records are written to:

logs/rejected_records.csv

The successful pipeline execution resulted in no rejected records.

The final analytical mart contains:

Total rows:       101,766
Unique encounters: 101,766
Unique patients:   71,518
11. Feature Engineering

The following analytical features were created during transformation.

patient_key

A patient-safe surrogate identifier generated from the source patient identifier.

The original patient identifier is not used directly in the analytical tables.

prior_visits

Calculated using:

number_inpatient
+ number_emergency
+ number_outpatient

This represents the patient's prior healthcare utilization.

diagnosis_count

Counts the available diagnosis fields:

diag_1
diag_2
diag_3
medication_count

Counts medication fields with meaningful medication activity such as:

Steady
Up
Down
age_group

The original age categories are converted into ordered age groups:

0-10
10-20
20-30
30-40
40-50
50-60
60-70
70-80
80-90
90-100

These features are stored in the cleaned dataset and analytical database.

12. Apache Airflow Orchestration

Apache Airflow is used to make the ETL pipeline repeatable and orchestrated.

DAG:

hospital_readmission_etl

Tasks:

ingest_data
      |
      v
transform_and_validate
      |
      v
load_postgresql

The DAG file is:

~/airflow/dags/hospital_readmission_etl.py

The DAG uses Python operators to execute the project ETL scripts.

The pipeline is configured with:

catchup = False
schedule = None

This allows the workflow to be manually triggered for reproducible execution during the Phase I demonstration.

A successful end-to-end execution was completed with all three tasks marked successful.

Observed execution duration:

Approximately 1 minute 6 seconds
13. Streamlit Dashboard

The interactive dashboard is implemented using Streamlit and Plotly.

Run:

streamlit run dashboard/app.py

The dashboard connects to PostgreSQL and retrieves analytical results using SQL queries.

Dashboard Indicators and Views

The dashboard provides more than five meaningful analytical views.

KPI Cards

The dashboard displays:

30-Day Readmission Rate
Total Encounters
30-Day Readmissions
Total Unique Patients

The overall 30-day readmission rate is approximately:

11.16%
Patient Profile

The Patient Profile section contains:

30-day readmission rate by age group
30-day readmission rate by gender
Readmission Analysis

The Readmission Analysis section contains:

Readmission rate by length of stay
Readmission rate by prior visits
Readmission rate by diagnosis count
Readmission rate by medical specialty
Readmission status distribution
Diagnosis Analysis

The Diagnosis Analysis section contains:

Readmission rate by primary diagnosis
Dataset Summary

The dashboard also provides:

Total encounters
Unique patients
30-day readmissions
Readmissions after 30 days
Non-readmitted encounters
Data Quality

The dashboard displays:

Total rows
Unique encounters
Unique patients
Data Export

Users can download the analytical readmission data as a CSV file.

14. Key Analytical Findings

The dashboard analysis produced the following observations.

Overall Readmission

There are:

101,766 total encounters
11,357 readmissions within 30 days
11.16% overall 30-day readmission rate
71,518 unique patients
Prior Healthcare Utilization

Readmission rates increase as prior healthcare utilization increases.

Prior Visits	Readmission Rate
0	8.18%
1-2	12.59%
3-5	16.39%
6-9	23.89%
10+	29.60%

This indicates that patients with a greater history of previous healthcare utilization have a higher observed readmission rate.

Age

The highest observed readmission rate among the age groups was:

20-30: 14.24%

Other relatively high groups include:

80-90: 12.08%
70-80: 11.77%
Length of Stay

Readmission rates generally increase across several higher length-of-stay categories.

Examples include:

1 day: 8.18%
5 days: 12.03%
8 days: 14.23%
10 days: 14.35%
Medical Specialty

Among specialties with at least 500 encounters, the highest observed readmission rates include:

Nephrology: 15.38%
Surgery-Vascular: 13.88%
Psychiatry: 12.18%
Family/GeneralPractice: 11.87%
Primary Diagnosis

Among sufficiently frequent primary diagnosis codes, several diagnosis categories show higher observed readmission rates.

The highest examples include:

250.7: 18.94%
250.6: 18.51%
434:   16.22%
440:   15.83%
820:   15.80%

These observations are descriptive and do not imply causal relationships.

15. Reproducibility

The project is designed so that the main ETL stages can be executed repeatedly.

Step 1 – Activate Environment
.\venv\Scripts\activate
Step 2 – Run Ingestion
python scripts/ingest.py
Step 3 – Run Transformation
python scripts/transform.py
Step 4 – Load PostgreSQL
python scripts/load_database.py
Step 5 – Start Dashboard
streamlit run dashboard/app.py
Step 6 – Run Airflow

In WSL:

source /mnt/e/Hospital_Readmission_Analytics/airflow-venv/bin/activate

Then use the Airflow UI to trigger:

hospital_readmission_etl

The Airflow DAG executes the same three ETL stages automatically:

Ingestion
    ↓
Transformation and Validation
    ↓
PostgreSQL Loading
16. Privacy and Security

The source dataset is de-identified and publicly available.

The project follows basic data-protection practices:

Raw data is retained separately from transformed data.
The original patient identifier is not used directly in analytical tables.
A surrogate patient_key is created.
Database credentials are stored in .env.
.env should not be committed to version control.
Validation rejects invalid records before analytical loading.
The dashboard exposes aggregated analytical information rather than direct patient identifiers.
17. Phase I Deliverables

The following Phase I requirements have been implemented:

 Public anonymized dataset acquired
 Raw data layer created
 Staging data layer created
 Cleaned data layer created
 Automated ingestion script
 Transformation and validation script
 Rejected-record/error log
 PostgreSQL database
 Analytical tables
 Readmission data mart
 Apache Airflow orchestration
 Successful end-to-end Airflow execution
 Streamlit dashboard
 More than five meaningful dashboard views/indicators
 Architecture documentation
 Data dictionary
 Validation rules
 Execution evidence
 README documentation

Phase II components such as machine-learning models, MLflow, FastAPI, Docker deployment, and monitoring are intentionally excluded from this Phase I implementation.

18. Demo Sequence

For demonstrating the project, the following sequence can be used:

1. Show the Source Dataset

Explain that the project uses the UCI Diabetes 130-US Hospitals dataset.

2. Show the Project Structure

Explain the raw, staging, cleaned, scripts, logs, and dashboard directories.

3. Show Data Ingestion

Run:

python scripts/ingest.py

Explain the creation of the raw/staging layers and ingestion metadata.

4. Show Transformation

Run:

python scripts/transform.py

Explain validation, rejected-record logging, cleaning, and feature engineering.

5. Show PostgreSQL

Run:

python scripts/load_database.py

Show the successful loading of the four analytical tables.

6. Show Airflow

Open the Airflow UI and show:

hospital_readmission_etl

Explain:

ingest_data
      ↓
transform_and_validate
      ↓
load_postgresql

Show the successful execution where all three tasks are green.

7. Show the Dashboard

Run:

streamlit run dashboard/app.py

Demonstrate:

KPI cards
Age analysis
Gender analysis
Length-of-stay analysis
Prior-visits analysis
Diagnosis-count analysis
Medical-specialty analysis
Readmission status
Primary diagnosis analysis
Dataset summary
Data quality
CSV export
19. References
UCI Machine Learning Repository – Diabetes 130-US Hospitals for Years 1999-2008.
Dataset ID: 296.
DOI: 10.24432/C5230J.
Apache Airflow Documentation.
PostgreSQL Documentation.
Streamlit Documentation.
Plotly Documentation.