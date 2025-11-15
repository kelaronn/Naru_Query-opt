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
        # Key columns slow down NARU extremelyly, so we skip them here.
        # 'L_ORDERKEY', 'L_PARTKEY', 'L_SUPPKEY',
        #'L_LINENUMBER',
        #'L_QUANTITY', 
        #'L_EXTENDEDPRICE', 
        #'L_DISCOUNT', 
        #'L_TAX',
        'L_RETURNFLAG', 
        'L_LINESTATUS', 
        'L_SHIPDATE', 
        'L_COMMITDATE',
        'L_RECEIPTDATE', 
        'L_SHIPINSTRUCT', 
        'L_SHIPMODE',
        # Comment column will never be indexed or filtered, so we can skip it.
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
