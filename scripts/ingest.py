import pandas as pd
import os
from datetime import datetime

# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

source_path = "data/raw/diabetic_data.csv"
staging_path = "data/staging/diabetic_data_staging.csv"


# --------------------------------------------------
# 2. Load raw data
# --------------------------------------------------

df = pd.read_csv(source_path)

print("Raw dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# --------------------------------------------------
# 3. Save a staging copy
# --------------------------------------------------

df.to_csv(
    staging_path,
    index=False
)

print("\nStaging dataset created successfully!")
print("Staging file:", staging_path)


# --------------------------------------------------
# 4. Create ingestion metadata
# --------------------------------------------------

metadata = {
    "extraction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "source_file": source_path,
    "status": "SUCCESS",
    "row_count": len(df),
    "column_count": len(df.columns)
}

metadata_df = pd.DataFrame([metadata])

metadata_df.to_csv(
    "data/staging/ingestion_metadata.csv",
    index=False
)

print("\nIngestion metadata:")
print(metadata_df)