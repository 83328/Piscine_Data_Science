import psycopg2
from dotenv import load_dotenv
import os
import plotext as plt
from datetime import datetime
import pandas as pd

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

    # Query purchase data (October 2022 to February 2023)
    query = """
        SELECT DATE_TRUNC('day', event_time) AS day, 
               COUNT(DISTINCT user_id) AS customer_count,
               SUM(price) AS total_price
        FROM customers
        WHERE event_type = 'purchase'
        AND event_time BETWEEN '2022-10-01' AND '2023-03-01'
        GROUP BY DATE_TRUNC('day', event_time)
        ORDER BY day;
    """
    cursor.execute(query)
    data = cursor.fetchall()

    # Check if data is returned
    if not data:
        raise Exception("No purchase data found for October 2022 to February 2023.")

    # Close connection
    cursor.close()
    conn.close()

    # Process data
    df = pd.DataFrame(data, columns=['day', 'customer_count', 'total_price'])
    day_labels = df['day'].dt.strftime('%d %b').tolist()  # Format as "01 Oct", "02 Oct", etc.
    customer_counts = df['customer_count'].astype(int).tolist()
    prices = df['total_price'].astype(float).tolist()
    x_indices = list(range(len(day_labels)))

    # Line Chart - Number of Customers (keep first)
    print("Line Chart:")
    plt.plot(x_indices, customer_counts, marker='dot', color='blue')

    # Extract just the month from each label
    month_labels = []
    current_month = None
    month_indices = []

    for i, label in enumerate(day_labels):
        month = label.split(' ')[1]  # Get just the month part (e.g., 'Oct')
        if month != current_month:
            month_labels.append(month)
            month_indices.append(i)
            current_month = month

    plt.xticks(month_indices, month_labels)  # Show month names at month transitions
    plt.title('Daily Customer Count (Line Chart)')
    plt.xlabel('Month')
    plt.ylabel('Number of Customers')
    plt.show()
    plt.clear_figure()

    # Bar Chart - Total Price (monthly data) - MOVE THIS TO SECOND POSITION
    print("Bar Chart:")

    # Create monthly aggregation for bar chart
    monthly_df = df.copy()
    monthly_df['month'] = monthly_df['day'].dt.strftime('%b %Y')  # Format as "Oct 2022"

    # Properly sort the months chronologically
    month_order = {'Oct 2022': 0, 'Nov 2022': 1, 'Dec 2022': 2, 'Jan 2023': 3, 'Feb 2023': 4}
    monthly_data = monthly_df.groupby('month').agg({'total_price': 'sum'}).reset_index()
    monthly_data['sort_order'] = monthly_data['month'].map(month_order)
    monthly_data = monthly_data.sort_values('sort_order').drop('sort_order', axis=1)

    # Convert decimal.Decimal to float before plotting
    monthly_data['total_price'] = monthly_data['total_price'].astype(float)

    # Plot monthly bar chart
    monthly_values = monthly_data['total_price'].tolist()
    month_labels = monthly_data['month'].tolist()
    short_month_labels = [m.split()[0] for m in month_labels]  # Just 'Oct', 'Nov', etc.

    # Clear only the figure settings, not the terminal
    plt.clear_figure()

    plt.bar(range(len(monthly_values)), monthly_values, color='blue', width=0.6)
    plt.xticks(range(len(monthly_values)), short_month_labels)  # Using shorter labels

    # Format Y-axis values for readability
    max_value = max(monthly_values)
    # Create 5 tick points from 0 to max_value
    y_ticks = [i * (max_value / 4) for i in range(5)]

    # Format as M for millions, K for thousands
    if max_value > 1000000:
        y_labels = [f"{y/1000000:.1f}M" if y > 0 else "0" for y in y_ticks]
    elif max_value > 1000:
        y_labels = [f"{y/1000:.1f}K" if y > 0 else "0" for y in y_ticks]
    else:
        y_labels = [str(int(y)) for y in y_ticks]

    plt.yticks(y_ticks, y_labels)

    plt.title('Monthly Purchase Totals (Bar Chart) - Altairian Dollars')
    plt.xlabel('Month')
    plt.ylabel('Total Price')
    plt.show()
    plt.clear_figure()

    # Area Chart - MOVE THIS TO LAST POSITION
    print("Area Chart:")

    # Calculate average spend per customer for each day
    average_spend = df['total_price'] / df['customer_count']
    average_spend = average_spend.astype(float).tolist()

    # Clear figure settings
    plt.clear_figure()

    # Create a solid filled area chart using smaller, denser markers
    # First plot all squares to create a filled effect
    min_value = min(average_spend)
    max_value = max(average_spend)

    # Use a much smaller step size
    vertical_step = max(0.1, max_value/100)  # Very small step for density
    horizontal_step = 1  # Use every x point

    # Fill the entire area with a denser pattern
    for x in range(0, len(x_indices), horizontal_step):
        # For each x position, fill from 0 up to the value
        curr_value = average_spend[x]
        for y in range(0, int(curr_value/vertical_step)):
            y_val = y * vertical_step
            if y_val <= curr_value:
                plt.scatter([x], [y_val], marker='4', color='blue')

    # Then plot the line on top to define the upper boundary
    plt.plot(x_indices, average_spend, marker='o', color='green')

    # Add month labels at transitions
    plt.xticks(month_indices, month_labels)

    # Format Y-axis for readability
    y_ticks = [i * (max_value / 4) for i in range(5)]
    if max_value > 1000:
        y_labels = [f"{y/1000:.1f}K" if y > 0 else "0" for y in y_ticks]
    else:
        y_labels = [f"{y:.1f}" if y > 0 else "0" for y in y_ticks]

    plt.yticks(y_ticks, y_labels)
    plt.title('Daily Average Spend per Customer (Area Chart) - Altairian Dollars')
    plt.xlabel('Month')
    plt.ylabel('Average Spend per Customer in Altairian Dollars')
    plt.show()

except Exception as e:
    print(f"Error: {e}")
    exit(1)