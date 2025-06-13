import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
db_params = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": "localhost",
    "port": 5432,
}

# Create SQLAlchemy engine
engine = create_engine(f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/{os.getenv('POSTGRES_DB')}")

# Connect and load relevant customer data
query = """
    SELECT user_id, SUM(price) as total_spent, COUNT(*) as frequency
    FROM customers
    WHERE event_type = 'purchase' AND price IS NOT NULL
    GROUP BY user_id
"""
df = pd.read_sql(query, engine)

# Preprocessing
features = df[["total_spent", "frequency"]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Elbow Method
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot
plt.figure(figsize=(6, 4))
plt.gca().set_facecolor("gainsboro")
plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.grid(True, color='white', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
