import psycopg2
from term_piechart import Pie
import os
from dotenv import load_dotenv

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
        raise Exception("Table 'customers' does not exist. Run 'make_db' in 1_Data_Warehouse first.")

    # Query event type distribution
    cursor.execute("SELECT event_type, COUNT(*) FROM customers GROUP BY event_type")
    events = [
        {
            "name": record[0],
            "value": record[1]
        }
        for record in cursor
    ]

    # Check if data is returned
    if not events:
        raise Exception("No data returned from customers table. Check contents.")

    # Close connection
    cursor.close()
    conn.close()

    # Create and render pie chart
    pie = Pie(
        events,
        radius=15,  # Smaller radius for alignment
        autocolor=True,
        autocolor_pastel_factor=0.1,
        fill='42',
        legend={"line": 1, "format": "{name:<8} {percent:>5.2f}%"}

    )
    print(pie)

except Exception as e:
    print(f"Error: {e}")
    exit(1)