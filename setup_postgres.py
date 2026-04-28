import pandas as pd
from sqlalchemy import create_engine
import os
import argparse

def setup_db(csv_path, db_url):
    print(f"Loading {csv_path} ...")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run generate_tpc-h_dataset.py first.")
        return False
        
    # Read the full dataset
    df = pd.read_csv(csv_path)
    
    # Convert dates properly for PostgreSQL
    date_cols = ['L_SHIPDATE', 'L_COMMITDATE', 'L_RECEIPTDATE']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            
    print(f"Connecting to database at {db_url}...")
    try:
        engine = create_engine(db_url)
        
        print(f"Writing {len(df)} rows to PostgreSQL table 'lineitem'...")
        # Write to database - this automatically creates the table schema
        df.to_sql('lineitem', engine, if_exists='replace', index=False)
        
        print("Data loaded successfully!")
        return True
    except Exception as e:
        print(f"Failed to connect or write to database: {e}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load CSV data into PostgreSQL')
    # Change the default values if needed
    parser.add_argument('--csv', type=str, default='datasets/tpch_lineitem_1m.csv', help='Path to CSV file')
    parser.add_argument('--db-url', type=str, default='postgresql://postgres:postgres@localhost:5432/naru_db', help='PostgreSQL connection string')
    
    args = parser.parse_args()
    setup_db(args.csv, args.db_url)
