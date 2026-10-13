import pandas as pd

# Path to the raw dataset
file_path = "data/raw/diabetic_data.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Display basic information
print("Dataset loaded successfully!")

print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

# Profile the special missing values
print("\nSpecial missing values:")
print((df == "?").sum().sort_values(ascending=False))

# Check all possible missing-value representations
print("\nCommon values in selected columns:")

print("\nRace:")
print(df["race"].value_counts(dropna=False))

print("\nMedical specialty:")
print(df["medical_specialty"].value_counts(dropna=False).head(10))

print("\nWeight:")
print(df["weight"].value_counts(dropna=False).head(10))

print("\nMax glucose serum:")
print(df["max_glu_serum"].value_counts(dropna=False))

print("\nA1C result:")
print(df["A1Cresult"].value_counts(dropna=False))

print("\nReadmission:")
print(df["readmitted"].value_counts())

# Check duplicates
print("\nDuplicate records:")
print("Number of duplicate rows:", df.duplicated().sum())

print("\nDuplicate encounter IDs:")
print("Number of duplicate encounter IDs:", df["encounter_id"].duplicated().sum())

print("\nDuplicate patient IDs:")
print("Number of unique patients:", df["patient_nbr"].nunique())

# Create a small profiling script
print("\nData validation checks:")

print("Minimum length of stay:", df["time_in_hospital"].min())
print("Maximum length of stay:", df["time_in_hospital"].max())

print("\nGender values:")
print(df["gender"].value_counts())

print("\nAge groups:")
print(df["age"].value_counts().sort_index())

print("\nDiagnosis count range:")
print(
    "Minimum:", df["number_diagnoses"].min(),
    "Maximum:", df["number_diagnoses"].max()
)

print("\nReadmission distribution:")
print(df["readmitted"].value_counts())