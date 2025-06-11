import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import os
from dotenv import load_dotenv

# --- Load environment variables from .env ---
load_dotenv()

user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")
db = os.getenv("POSTGRES_DB")
host = 'localhost'
port = 5432

# --- Connect to PostgreSQL ---
url = f'postgresql://{user}:{password}@{host}:{port}/{db}'
engine = create_engine(url)

# --- Loop through all CSV files in customer/ directory ---
csv_dir = 'customer'

for filename in os.listdir(csv_dir):
    if filename.endswith('.csv'):
        csv_path = os.path.join(csv_dir, filename)
        table_name = os.path.splitext(filename)[0]  # → e.g. 'data_2022_oct'

        df = pd.read_csv(csv_path)

        print(f"\n📌 Processing '{filename}' → Table: '{table_name}'")
        print("🧾 Columns:", list(df.columns))

        # --- Add required/synthetic columns ---
        df.insert(0, 'created_at', datetime.now())  # TIMESTAMP
        df['id'] = range(1, len(df) + 1)             # INTEGER

        # --- Type conversions (6+ types) ---
        df['created_at'] = pd.to_datetime(df['created_at'])       # TIMESTAMP
        if 'product_id' in df.columns:
            df['product_id'] = df['product_id'].astype(str)       # TEXT / VARCHAR
        if 'price' in df.columns:
            df['price'] = df['price'].astype(float)               # FLOAT
        if 'user_id' in df.columns:
            df['user_id'] = df['user_id'].astype(int)             # INTEGER
        if 'user_session' in df.columns:
            df['user_session'] = df['user_session'].astype(str)   # TEXT
        if 'event_type' in df.columns:
            df['event_type'] = df['event_type'].astype('category') # CATEGORICAL

        # --- Create or replace table in PostgreSQL ---
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"✅ Uploaded '{table_name}' to DB.")
