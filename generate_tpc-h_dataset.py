import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_tpch_lineitem(num_rows):
    """
    Generates a TPC-H 'LINEITEM' style dataset.
    """
    print(f"Generating data with {num_rows} rows...")

    # Helper function to generate random dates within a range
    def random_dates(start, end, n):
        start_u = start.value // 10**9
        end_u = end.value // 10**9
        return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')

    # Set base date range (1992-1998 is the standard TPC-H interval)
    start_date = pd.to_datetime('1992-01-01')
    end_date = pd.to_datetime('1998-12-31')

    # Generate columns
    data = {
        'L_ORDERKEY': np.random.randint(1, num_rows * 2, size=num_rows),
        'L_PARTKEY': np.random.randint(1, 2000, size=num_rows),
        'L_SUPPKEY': np.random.randint(1, 100, size=num_rows),
        'L_LINENUMBER': np.random.randint(1, 8, size=num_rows),
        'L_QUANTITY': np.random.randint(1, 50, size=num_rows), 
        'L_EXTENDEDPRICE': np.random.randint(1, 1000, size=num_rows),
        'L_DISCOUNT': np.random.choice([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10], size=num_rows),
        'L_TAX': np.random.choice([0.00, 0.01, 0.02, 0.04, 0.06, 0.08], size=num_rows),
        'L_RETURNFLAG': np.random.choice(['R', 'A', 'N'], size=num_rows), # Flag: Returned, Assembled, None
        'L_LINESTATUS': np.random.choice(['O', 'F'], size=num_rows), # Status: Open, Filled
        'L_SHIPDATE': random_dates(start_date, end_date, num_rows),
    }

    # COMMITDATE and RECEIPTDATE depend on SHIPDATE
    # Commit date = ship date + random days (e.g., 10-30 days)
    # Receipt date = ship date + random days (e.g., 1-30 days)
    ship_dates = data['L_SHIPDATE']
    data['L_COMMITDATE'] = ship_dates + pd.to_timedelta(np.random.randint(10, 60, size=num_rows), unit='D')
    data['L_RECEIPTDATE'] = ship_dates + pd.to_timedelta(np.random.randint(1, 30, size=num_rows), unit='D')

    # Text columns (fixed length codes according to standard)
    instructions = ['DELIVER IN PERSON', 'COLLECT COD', 'NONE', 'TAKE BACK RETURN']
    modes = ['RAIL', 'SHIP', 'AIR', 'TRUCK', 'MAIL', 'FOB', 'REG AIR']

    data['L_SHIPINSTRUCT'] = np.random.choice(instructions, size=num_rows)
    data['L_SHIPMODE'] = np.random.choice(modes, size=num_rows)
    
    # Generate comments (random text)
    # Simplified for performance
    comments = [
        "regular customer request", "urgent delivery needed", 
        "check packaging carefully", "fragile items inside", 
        "delayed by supplier", "customer compliant regarding quality"
    ]
    data['L_COMMENT'] = np.random.choice(comments, size=num_rows)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Format dates to string (optional, but looks better in CSV)
    date_cols = ['L_SHIPDATE', 'L_COMMITDATE', 'L_RECEIPTDATE']
    for col in date_cols:
        df[col] = df[col].dt.strftime('%Y-%m-%d')

    return df

if __name__ == "__main__":
    print(">>> SCRIPT STARTED...")
    
    # Generate 10,000 rows
    ROW_COUNT = 10000
    df_tpch = generate_tpch_lineitem(ROW_COUNT)
    
    # Save to CSV
    filename = "tpch_lineitem_10k.csv"
    df_tpch.to_csv(filename, index=False)
    
    print(f"Done! File saved: {filename}")
    print(f"Number of columns: {len(df_tpch.columns)}")
    print("First 5 rows:")
    print(df_tpch.head())