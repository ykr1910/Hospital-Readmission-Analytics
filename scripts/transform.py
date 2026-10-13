import pandas as pd

# --------------------------------------------------
# 1. Load raw data
# --------------------------------------------------

file_path = "data/staging/diabetic_data_staging.csv"

df = pd.read_csv(file_path)

print("Raw data loaded successfully!")
print("Original rows:", len(df))
print("Original columns:", len(df.columns))

# --------------------------------------------------
# 2. Data quality validation
# --------------------------------------------------

rejected_records = []

# Check 1: Missing encounter ID
invalid_encounter_id = df["encounter_id"].isna()

for index in df[invalid_encounter_id].index:
    rejected_records.append({
        "row_number": index,
        "reason": "Missing encounter_id"
    })


# Check 2: Duplicate encounter ID
duplicate_encounter_id = df["encounter_id"].duplicated(keep=False)

for index in df[duplicate_encounter_id].index:
    rejected_records.append({
        "row_number": index,
        "reason": "Duplicate encounter_id"
    })


# Check 3: Invalid length of stay
invalid_length_of_stay = (
    (df["time_in_hospital"] < 1) |
    (df["time_in_hospital"] > 14)
)

for index in df[invalid_length_of_stay].index:
    rejected_records.append({
        "row_number": index,
        "reason": "Invalid time_in_hospital"
    })


# Check 4: Invalid diagnosis count
invalid_diagnosis_count = (
    (df["number_diagnoses"] < 1) |
    (df["number_diagnoses"] > 16)
)

for index in df[invalid_diagnosis_count].index:
    rejected_records.append({
        "row_number": index,
        "reason": "Invalid number_diagnoses"
    })


# Check 5: Missing readmission status
invalid_readmission = df["readmitted"].isna()

for index in df[invalid_readmission].index:
    rejected_records.append({
        "row_number": index,
        "reason": "Missing readmission status"
    })


# Save rejected records
rejected_df = pd.DataFrame(
    rejected_records,
    columns=["row_number", "reason"]
)

if len(rejected_df) > 0:
    rejected_df.to_csv(
        "logs/rejected_records.csv",
        index=False
    )
else:
    # Create an empty rejected-record log
    # with the correct column names
    rejected_df.to_csv(
        "logs/rejected_records.csv",
        index=False
    )

print("\nData quality validation completed.")
print("Rejected records:", len(rejected_df))

# Remove invalid records from the dataset
invalid_rows = set(
    rejected_df["row_number"].tolist()
)

df = df.drop(index=invalid_rows)

print("Rows remaining after validation:", len(df))

# --------------------------------------------------
# 2. Replace special missing-value codes
# --------------------------------------------------

df = df.replace("?", pd.NA)

print("\nSpecial missing values replaced.")


# --------------------------------------------------
# 3. Handle selected missing values
# --------------------------------------------------

# Weight has approximately 97% missing values,
# so it is excluded from the analytical dataset.
df = df.drop(columns=["weight"])

# For categorical fields, preserve the record
# and explicitly label missing values as Unknown.
df["race"] = df["race"].fillna("Unknown")
df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")
df["payer_code"] = df["payer_code"].fillna("Unknown")

# Gender has 3 Unknown/Invalid values.
df["gender"] = df["gender"].replace(
    "Unknown/Invalid", "Unknown"
)


# --------------------------------------------------
# 4. Create patient-safe surrogate key
# --------------------------------------------------

patient_mapping = (
    df[["patient_nbr"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

patient_mapping["patient_key"] = (
    "P" +
    (patient_mapping.index + 1)
    .astype(str)
    .str.zfill(6)
)

df = df.merge(
    patient_mapping,
    on="patient_nbr",
    how="left"
)


# --------------------------------------------------
# 5. Create engineered variables
# --------------------------------------------------

# Total previous visits
df["prior_visits"] = (
    df["number_inpatient"]
    + df["number_emergency"]
    + df["number_outpatient"]
)

# Diagnosis count based on available diagnosis fields
diagnosis_columns = ["diag_1", "diag_2", "diag_3"]

df["diagnosis_count"] = (
    df[diagnosis_columns]
    .notna()
    .sum(axis=1)
)

# Medication count
medication_columns = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone"
]

df["medication_count"] = (
    df[medication_columns]
    .isin(["Steady", "Up", "Down"])
    .sum(axis=1)
)


# --------------------------------------------------
# 6. Create age groups
# --------------------------------------------------

df["age_group"] = df["age"].str.replace(
    "[", "", regex=False
).str.replace(
    ")", "", regex=False
)


# --------------------------------------------------
# 7. Display transformed data
# --------------------------------------------------

print("\nTransformation completed!")

print("Rows after transformation:", len(df))
print("Columns after transformation:", len(df.columns))

print("\nNew columns:")
print([
    "patient_key",
    "prior_visits",
    "diagnosis_count",
    "medication_count",
    "age_group"
])

print("\nSample transformed data:")
print(
    df[
        [
            "encounter_id",
            "patient_key",
            "age_group",
            "time_in_hospital",
            "prior_visits",
            "diagnosis_count",
            "medication_count",
            "readmitted"
        ]
    ].head()
)

# --------------------------------------------------
# 8. Save cleaned dataset
# --------------------------------------------------

output_path = "data/cleaned/cleaned_hospital_data.csv"

df.to_csv(
    output_path,
    index=False
)

print("\nCleaned dataset saved successfully!")
print("Output file:", output_path)