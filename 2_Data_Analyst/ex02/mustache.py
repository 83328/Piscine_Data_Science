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
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    # Get individual purchase prices
    cursor.execute("""
        SELECT price FROM customers
        WHERE event_type = 'purchase' AND price IS NOT NULL;
    """)
    price_data = cursor.fetchall()

    # Get average basket value per user
    cursor.execute("""
        WITH basket_contents AS (
            SELECT user_id,
                   SUM(CASE 
                       WHEN event_type = 'cart' THEN price
                       WHEN event_type = 'remove_from_cart' THEN -price
                       ELSE 0 END) AS basket_total
            FROM customers
            WHERE event_type IN ('cart', 'remove_from_cart') AND price IS NOT NULL
            GROUP BY user_id
            HAVING SUM(CASE 
                       WHEN event_type = 'cart' THEN price
                       WHEN event_type = 'remove_from_cart' THEN -price
                       ELSE 0 END) > 0
        )
        SELECT basket_total FROM basket_contents;
    """)
    basket_data = cursor.fetchall()

    cursor.close()
    conn.close()

    # Convert to DataFrames and cast to float
    df_prices = pd.DataFrame(price_data, columns=['price'])
    df_prices['price'] = df_prices['price'].astype(float)

    df_baskets = pd.DataFrame(basket_data, columns=['avg_basket'])
    df_baskets['avg_basket'] = df_baskets['avg_basket'].astype(float)

    # Print statistics
    print("Purchase Price Statistics:")
    stats = df_prices['price'].describe()
    print(f"count      {stats['count']:.6f}")
    print(f"mean       {stats['mean']:.6f}")
    print(f"std        {stats['std']:.6f}")
    print(f"min        {stats['min']:.6f}")
    print(f"25%        {stats['25%']:.6f}")
    print(f"50%        {stats['50%']:.6f}")
    print(f"75%        {stats['75%']:.6f}")
    print(f"max        {stats['max']:.6f}")
    print()

    # -----------------------
    # 1. Horizontal Box Plot (raw purchase prices)
    plt.figure()
    plt.boxplot(df_prices['price'], vert=False, widths=0.9,
                patch_artist=True,  # Enable filling boxes
                boxprops=dict(facecolor='dimgray', edgecolor='none'),  # Fill with dimgray, no outline
                flierprops={'marker': 'D', 'markerfacecolor': 'dimgray', 'markeredgecolor': 'none', 'markersize': 4},  # No outline on markers
                medianprops={'color': 'white'},  # Make median line visible against gray background
                whiskerprops={'color': 'dimgray'},
                capprops={'color': 'dimgray'})
    plt.title("Box Plot of Individual Purchase Prices")
    plt.xlabel("price (₳)")
    plt.xlim(-100, 350)
    plt.yticks([])  # Remove Y-axis labels/ticks
    plt.grid(True, color='white', linestyle='-', linewidth=0.5)  # White grid lines
    plt.gca().set_facecolor('gainsboro')  # Only set the axes area to light gray
    plt.show()

    # -----------------------
    # === 2. Second Plot: Boxplot excluding outliers (IQR range only) ===
    q1 = df_prices['price'].quantile(0.25)
    q3 = df_prices['price'].quantile(0.75)
    iqr_df = df_prices[(df_prices['price'] >= q1) & (df_prices['price'] <= q3)]

    # Create a taller figure
    plt.figure(figsize=(10, 5))  # Remove the facecolor parameter here too

    # Make box taller and dark sea green
    plt.boxplot(iqr_df['price'], vert=False, 
                patch_artist=True,
                boxprops=dict(facecolor='darkseagreen'),
                widths=0.9)  # Make the box taller (0.5 is default)

    plt.title("Box plot (without outliers)")
    plt.xlabel("price (₳)")
    plt.yticks([])  # Remove Y-axis labels/ticks
    plt.grid(True, color='white', linestyle='-', linewidth=0.5)  # White grid lines
    plt.gca().set_facecolor('gainsboro')  # Only set the axes area to light gray
    plt.show()

    #=== 3. Third Plot: Average basket price per user ===
    # Use the basket values directly without grouping
    basket_values = df_baskets['avg_basket'].tolist()

    plt.figure()
    plt.boxplot(basket_values, vert=False, patch_artist=True,
                boxprops=dict(facecolor='cornflowerblue'),
                showfliers=True,  # Show outliers beyond the whiskers
                flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': 'red'},  # Remove the extra )
                widths=0.9)  # Add the width parameter here
    plt.title("Average basket price per user")
    plt.xlabel("price (₳)")
    plt.grid(True, color='white', linestyle='-', linewidth=0.5)
    plt.gca().set_facecolor('gainsboro')

    # Get the actual min/max values
    q1 = np.percentile(basket_values, 25)
    q3 = np.percentile(basket_values, 75)
    iqr = q3 - q1
    whisker_min = max(0, q1 - 1.5 * iqr)  # Don't go below 0
    whisker_max = min(80, q3 + 1.5 * iqr)  # Don't go above 80

    plt.xlim(whisker_min, whisker_max)
    plt.yticks([])  # Remove Y-axis labels/ticks
    plt.show()

except Exception as e:
    print("Error:", e)
    exit(1)
