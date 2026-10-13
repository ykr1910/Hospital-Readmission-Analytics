import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load variables from .env
load_dotenv()

# Read database settings
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Build database connection URL
DATABASE_URL = (
    f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Create database engine
engine = create_engine(DATABASE_URL)

# Test connection
try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))

        print("Database connection successful!")
        print("Test result:", result.scalar())

except Exception as e:
    print("Database connection failed.")
    print("Error:", e)