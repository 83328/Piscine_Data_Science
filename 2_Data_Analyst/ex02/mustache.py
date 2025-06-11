import psycopg2
from dotenv import load_dotenv
import os
import plotext as plt
import pandas as pd
import numpy as np

# Load .env file
env_path = '../.env'
load_dotenv(env_path)

# Database connection parameters
db_params = {
    'host': 'localhost',
    'dbname': os.getenv('POSTGRES_DB'),
    'user': os.getenv('POSTGRES_USER'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'port': '5432'
}

try:
    # Connect to PostgreSQL
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    # Verify customers table exists
    cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'customers');")
    if not cursor.fetchone()[0]:
        raise Exception("Table 'customers' does not exist. Run 'make ex01' in 1_Data_Warehouse first.")

    # Query purchase prices
    query = """
        SELECT price
        FROM customers
        WHERE event_type = 'purchase' AND price IS NOT NULL;
    """
    cursor.execute(query)
    data = cursor.fetchall()

    # Check if data is returned
    if not data:
        raise Exception("No valid purchase data found.")

    # Close connection
    cursor.close()
    conn.close()

    # Process data
    df = pd.DataFrame(data, columns=['price'])
    df['price'] = pd.to_numeric(df['price'], errors='coerce')  # Convert to numeric, handle non-numeric
    df = df.dropna()  # Remove NaN values
    if df.empty:
        raise Exception("No valid numeric price data after cleaning.")

    stats = df['price'].describe()

    # Print statistics
    print("Price Statistics for Purchased Items:")
    print(f"count    {stats['count']:.6f}")
    print(f"mean     {stats['mean']:.6f}")
    print(f"std      {stats['std']:.6f}")
    print(f"min      {stats['min']:.6f}")
    print(f"25%      {stats['25%']:.6f}")
    print(f"50%      {stats['50%']:.6f}")
    print(f"75%      {stats['75%']:.6f}")
    print(f"max      {stats['max']:.6f}")

    # Save statistics to CSV
    stats_df = pd.DataFrame({
        'metric': ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],
        'value': [stats['count'], stats['mean'], stats['std'], stats['min'], stats['25%'], stats['50%'], stats['75%'], stats['max']]
    })
    stats_df.to_csv('price_stats.csv', index=False)

    # Box Plot
    plt.clear_terminal()
    print("Box Plot:")
    plt.box(df['price'].tolist(), orientation='vertical')
    plt.title('Price Distribution of Purchased Items (Box Plot)')
    plt.ylabel('Price in Dollars')
    plt.show()

except Exception as e:
    print(f"Error: {e}")
    exit(1)