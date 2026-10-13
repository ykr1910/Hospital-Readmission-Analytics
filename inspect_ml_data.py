
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

url = (
    f"postgresql+psycopg://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}@127.0.0.1:"
    f"{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

engine = create_engine(url)

with engine.connect() as conn:
    columns = conn.execute(text("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'healthcare'
          AND table_name = 'readmission_mart'
        ORDER BY ordinal_position
    """)).fetchall()

    print("Columns in healthcare.readmission_mart:")
    for column, data_type in columns:
        print(f"{column}: {data_type}")

engine.dispose()
