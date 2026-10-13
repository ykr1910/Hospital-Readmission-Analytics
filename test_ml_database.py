
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DB_HOST = "127.0.0.1"
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DATABASE_URL = (
    f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)

try:
    with engine.connect() as conn:
        total = conn.execute(
            text("SELECT COUNT(*) FROM healthcare.readmission_mart")
        ).scalar()

        labels = conn.execute(
            text("""
                SELECT readmitted, COUNT(*)
                FROM healthcare.readmission_mart
                GROUP BY readmitted
                ORDER BY readmitted
            """)
        ).fetchall()

        print("Total encounters:", total)
        print("Readmission label counts:")
        for label, count in labels:
            print(f"{label}: {count}")

except Exception as e:
    print("Database connection failed:", e)

finally:
    engine.dispose()
