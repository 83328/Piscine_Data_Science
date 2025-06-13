import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from decimal import Decimal

# Load environment variables
load_dotenv('../.env')

# Connect to PostgreSQL
db_params = {
    'host': 'localhost',
    'dbname': os.getenv('POSTGRES_DB'),
    'user': os.getenv('POSTGRES_USER'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'port': 5432,
}

try:
    # Connect to PostgreSQL
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    # Verify customers table exists
    cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'customers');")
    if not cursor.fetchone()[0]:
        raise Exception("Table 'customers' does not exist. Run 'make ex01' in 1_Data_Warehouse first.")

    # Query order frequency and total spending per customer
    query = """
        SELECT 
            user_id,
            COUNT(*) AS order_frequency,
            COALESCE(SUM(price), 0) AS total_spending
        FROM customers
        WHERE event_type = 'purchase'
        GROUP BY user_id;
    """
    cursor.execute(query)
    data = cursor.fetchall()

    # Check if data is returned
    if not data:
        raise Exception("No purchase data found.")

    # Close connection
    cursor.close()
    conn.close()

    # Process data
    df = pd.DataFrame(data, columns=['user_id', 'order_frequency', 'total_spending'])
    
    # Bin order frequency (0-10, 10-20, 20-30, 30+)
    bins_freq = [0, 10, 20, 30, float('inf')]
    labels_freq = ['0-10', '10-20', '20-30', '30+']
    df_freq = pd.cut(df['order_frequency'], bins=bins_freq, labels=labels_freq, right=False)
    freq_counts = df_freq.value_counts().sort_index()

    # Bin total spending (0-50, 50-100, 100-150, 150+)
    bins_spend = [0, 50, 100, 150, float('inf')]
    labels_spend = ['0-50', '50-100', '100-150', '150+']
    df_spend = pd.cut(df['total_spending'], bins=bins_spend, labels=labels_spend, right=False)
    spend_counts = df_spend.value_counts().sort_index()

    # Create a single figure with two subplots side by side
    plt.figure(figsize=(14, 6))

    # First subplot: Order Frequency
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, position 1
    ax1 = plt.gca()
    ax1.set_facecolor('gainsboro')  # Set background to gainsboro
    plt.bar(freq_counts.index, freq_counts.values, color='cornflowerblue')
    plt.title('Number of Orders According to Frequency')
    plt.xlabel('Order Frequency Range')
    plt.ylabel('Number of Customers')
    plt.grid(True, color='white', linestyle='-', linewidth=0.5)  # White grid lines

    # Second subplot: Altairian Dollars Spent
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, position 2
    ax2 = plt.gca()
    ax2.set_facecolor('gainsboro')  # Set background to gainsboro
    plt.bar(spend_counts.index, spend_counts.values, color='cornflowerblue')
    plt.title('Altairian Dollars Spent by Customers')
    plt.xlabel('Monetary Value Range (₳)')
    plt.ylabel('Number of Customers')
    plt.grid(True, color='white', linestyle='-', linewidth=0.5)  # White grid lines

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the figure with both charts
    plt.show()

except Exception as e:
    print(f"Error: {e}")
    exit(1)