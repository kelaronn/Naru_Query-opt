"""Dataset registrations."""
import os

import numpy as np

import common


def LoadDmv(filename='Vehicle__Snowmobile__and_Boat_Registrations.csv'):
    csv_file = './datasets/{}'.format(filename)
    cols = [
        'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
        'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
        'Suspension Indicator', 'Revocation Indicator'
    ]
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'Reg Valid Date': np.datetime64}
    return common.CsvTable('DMV', csv_file, cols, type_casts)

def LoadTpch(filename='tpch_lineitem_10k.csv'):
    # Path to the file. If you didn't save it in the 'datasets' folder,
    # change it to './{}'.format(filename) or move the csv to 'datasets'.
    csv_file = './datasets/{}'.format(filename)

    cols = [
        # Key columns slow down NARU extremely, so we skip them here.
        # 'L_ORDERKEY', 'L_PARTKEY', 'L_SUPPKEY',
        'L_LINENUMBER',
        'L_QUANTITY', 
        #'L_EXTENDEDPRICE', 
        'L_DISCOUNT', 
        'L_TAX',
        'L_RETURNFLAG', 
        'L_LINESTATUS', 
        'L_SHIPDATE', 
        'L_COMMITDATE',
        'L_RECEIPTDATE', 
        'L_SHIPINSTRUCT', 
        'L_SHIPMODE',
        # Comment column is unlikely to be indexed or filtered, so we can skip it.
        # 'L_COMMENT'
    ]

    # Note: Numerical columns are usually handled automatically by the system.
    # However, dates must be explicitly converted to np.datetime64 to ensure
    # that filtering (range queries) works correctly.
    type_casts = {
        'L_SHIPDATE': np.datetime64,
        'L_COMMITDATE': np.datetime64,
        'L_RECEIPTDATE': np.datetime64,
    }
    return common.CsvTable('TPCH_LINEITEM', csv_file, cols, type_casts)

def LoadTpchFromPostgres(db_url='postgresql://postgres:postgres@localhost:5432/naru_db', table_name='lineitem'):
    """Loads TPC-H data directly from a PostgreSQL database using pandas."""
    import pandas as pd
    from sqlalchemy import create_engine
    
    engine = create_engine(db_url)
    cols = [
        'L_LINENUMBER',
        'L_QUANTITY', 
        'L_DISCOUNT', 
        'L_TAX',
        'L_RETURNFLAG', 
        'L_LINESTATUS', 
        'L_SHIPDATE', 
        'L_COMMITDATE',
        'L_RECEIPTDATE', 
        'L_SHIPINSTRUCT', 
        'L_SHIPMODE',
    ]
    
    print(f"Loading '{table_name}' from PostgreSQL...")
    df = pd.read_sql_table(table_name, engine, columns=cols)
    
    type_casts = {
        'L_SHIPDATE': np.datetime64,
        'L_COMMITDATE': np.datetime64,
        'L_RECEIPTDATE': np.datetime64,
    }
    
    # We can reuse CsvTable because it accepts a DataFrame instead of a filename
    return common.CsvTable('TPCH_LINEITEM_PG', df, cols=None, type_casts=type_casts)
