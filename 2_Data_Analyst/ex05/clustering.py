import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('../.env')

# Create SQLAlchemy engine
engine = create_engine(f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/{os.getenv('POSTGRES_DB')}")

# Query for RFM analysis - does everything in SQL
query = """
    WITH purchase_data AS (
        SELECT 
            user_id,
            event_time,
            price
        FROM customers
        WHERE event_type = 'purchase' AND price IS NOT NULL
    ),
    max_date AS (
        SELECT MAX(event_time) as max_date FROM purchase_data
    ),
    rfm AS (
        SELECT 
            user_id,
            COUNT(*) as frequency,
            SUM(price) as monetary_value,
            EXTRACT(DAY FROM (
                (SELECT max_date FROM max_date) - MAX(event_time)
            )) as recency
        FROM purchase_data
        GROUP BY user_id
    )
    SELECT * FROM rfm;
"""

# Load data directly from the database
grouped = pd.read_sql(query, engine)

# Scale features for better clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(grouped[['frequency', 'monetary_value', 'recency']])

# Apply KMeans clustering with 5 clusters
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
grouped["cluster"] = kmeans.fit_predict(scaled_features)

# Analyze cluster centers to name them
centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_info = pd.DataFrame(centers, columns=['frequency', 'monetary_value', 'recency'])

# Name clusters based on their characteristics
cluster_names = {
    # These will be determined after looking at centers
    # Example: cluster with high frequency, high monetary value, low recency = "Platinum"
}

# Determine cluster names by characteristics
for i in range(n_clusters):
    freq = cluster_info.iloc[i]['frequency']
    monetary = cluster_info.iloc[i]['monetary_value']
    recency = cluster_info.iloc[i]['recency']
    
    if freq > cluster_info['frequency'].median() and monetary > cluster_info['monetary_value'].median():
        if recency < cluster_info['recency'].median():
            cluster_names[i] = "Platinum Members"
        else:
            cluster_names[i] = "Gold Members"
    elif freq < cluster_info['frequency'].median() and recency > cluster_info['recency'].quantile(0.75):
        cluster_names[i] = "Inactive Customers"
    elif freq < cluster_info['frequency'].median() and recency < cluster_info['recency'].quantile(0.25):
        cluster_names[i] = "New Customers"
    else:
        cluster_names[i] = "Silver Members"

# Add names to dataframe
grouped['segment'] = grouped['cluster'].map(cluster_names)

# First, calculate average purchase price for each customer
grouped['avg_purchase_price'] = grouped['monetary_value'] / grouped['frequency']

# VISUALIZATION 1: Scatter plot with labeled clusters
plt.figure(figsize=(12, 8))
plt.gca().set_facecolor('gainsboro')

colors = cm.viridis(np.linspace(0, 1, n_clusters))
for i, (cluster_id, name) in enumerate(cluster_names.items()):
    cluster_data = grouped[grouped["cluster"] == cluster_id]
    plt.scatter(
        cluster_data["frequency"], 
        cluster_data["monetary_value"], 
        color=colors[i],
        alpha=0.7,
        s=60,
        label=f"{name} (n={len(cluster_data)})"
    )

# Add cluster centers
plt.scatter(
    cluster_info['frequency'], 
    cluster_info['monetary_value'], 
    s=100, 
    marker='X', 
    color='red', 
    label='Segment Centers'
)

plt.xlabel("Total purchases", fontsize=12)
plt.ylabel("Total Spent (Altairian Dollars)", fontsize=12)
plt.title("Customer Segments for Targeted Marketing", fontsize=16)
plt.legend(fontsize=10)
plt.grid(True, color='white', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.show()

# VISUALIZATION 2: Bar chart of segment sizes
plt.figure(figsize=(10, 6))
plt.gca().set_facecolor('gainsboro')

segment_counts = grouped['segment'].value_counts().sort_values(ascending=False)
plt.bar(
    segment_counts.index, 
    segment_counts.values,
    color=colors[:len(segment_counts)]
)

plt.title("Size of Customer Segments", fontsize=16)
plt.xlabel("Customer Segment", fontsize=12)
plt.ylabel("Number of Customers", fontsize=12)
plt.grid(True, color='white', linestyle='-', linewidth=0.5)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# # VISUALIZATION 3: Average purchase price per segment
# plt.figure(figsize=(12, 8))
# plt.gca().set_facecolor('gainsboro')

# # Create box plots for each segment
# segment_order = ['Platinum Members', 'Gold Members', 'Silver Members', 'New Customers', 'Inactive Customers']
# segment_data = [grouped[grouped['segment'] == segment]['avg_purchase_price'] for segment in segment_order]

# # Create the box plot
# box = plt.boxplot(segment_data, patch_artist=True, vert=False, labels=segment_order)

# # Customize box colors
# for i, patch in enumerate(box['boxes']):
#     patch.set_facecolor(colors[i])

# plt.title('Average Purchase Price by Customer Segment', fontsize=16)
# plt.xlabel('Average Price per Purchase (Altairian Dollars)', fontsize=12)
# plt.ylabel('Customer Segment', fontsize=12)
# plt.grid(True, color='white', linestyle='-', linewidth=0.5)
# plt.tight_layout()
# plt.show()

# # VISUALIZATION 4: Bar chart of average purchase price
# plt.figure(figsize=(10, 6))
# plt.gca().set_facecolor('gainsboro')

# # Calculate mean avg_purchase_price for each segment
# segment_avg_price = grouped.groupby('segment')['avg_purchase_price'].mean().reindex(segment_order)

# # Create bar chart
# bars = plt.bar(
#     segment_avg_price.index,
#     segment_avg_price.values,
#     color=colors[:len(segment_avg_price)]
# )

# # Add value labels on top of bars
# for bar in bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
#              f'₳{height:.2f}', ha='center', fontsize=9)

# plt.title('Average Purchase Value by Customer Segment', fontsize=16)
# plt.xlabel('Customer Segment', fontsize=12)
# plt.ylabel('Average Purchase Price (₳)', fontsize=12)
# plt.grid(True, color='white', linestyle='-', linewidth=0.5)
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()
