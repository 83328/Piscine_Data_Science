import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import os
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")
db = os.getenv("POSTGRES_DB")
host = 'localhost'
port = 5432

# --- Load CSV ---
csv_path = 'item/item.csv'
table_name = 'items'
df = pd.read_csv(csv_path)

print("📌 CSV Columns:", list(df.columns))

# --- Add required/synthetic columns ---
df.insert(0, 'created_at', datetime.now())  # TIMESTAMP
df['id'] = range(1, len(df) + 1)             # INTEGER

# --- Type conversions (3+ types) ---
df['created_at'] = pd.to_datetime(df['created_at'])  # TIMESTAMP
df['id'] = df['id'].astype(int)                      # INTEGER

# You can adjust column types depending on the CSV content.
# For example, assuming the CSV has: item_id, name, price
if 'price' in df.columns:
    df['price'] = df['price'].astype(float)          # FLOAT
if 'name' in df.columns:
    df['name'] = df['name'].astype(str)              # TEXT

# --- PostgreSQL connection ---
url = f'postgresql://{user}:{password}@{host}:{port}/{db}'
engine = create_engine(url)

# --- Upload to PostgreSQL ---
print(f"🚀 Uploading table '{table_name}' to DB '{db}'...")
df.to_sql(table_name, engine, if_exists='replace', index=False)
print("✅ Table upload complete.")
