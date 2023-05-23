print("script started")
import os
import argparse
import os.path
import tables as tb

cwd = os.getcwd()
import sys

sys.path.append("..")

import third_party_libraries.pandas_to_postgres
from third_party_libraries.pandas_to_postgres import DataFrameCopy, hdf_to_postgres
import third_party_libraries.python_binance.binance
from third_party_libraries.python_binance.binance.client import Client
from third_party_libraries.python_binance.binance.enums import (
    KLINE_INTERVAL_1MINUTE,
    WEBSOCKET_DEPTH_20,
)
from third_party_libraries.python_binance.binance.websockets import BinanceSocketManager
from library.core import *
import ast
import pandas as pd
import numpy as np
import csv
import time
from io import StringIO

import sqlalchemy
from sqlalchemy import create_engine, event
from sqlalchemy.dialects import postgresql
from sqlalchemy import MetaData, Table

from support_files.locations import (
    binance_apiKey as binKey,
    binance_apiSecret as binSecret,
    SaveDir,
    DB_HOST,
    DB_PASSWORD,
    DB_USER,
    DB_NAME,
)

# Sorting out whether we are using the ipython kernel or not
try:
    get_ipython()
    check_if_ipython = True
    orderbook_path = "../data/orderbook/"
except Exception:
    check_if_ipython = False
    split_cwd = cwd.split("/")
    last_string = split_cwd.pop(-1)
    cwd = cwd.replace(last_string, "")
    os.chdir(cwd)
    parser = argparse.ArgumentParser(description="Binance Socket API")
    parser.add_argument("-s", "--symbol", type=str, help="symbol")
    args = parser.parse_args()

    if args.symbol != None:
        depth_table = args.symbol

    orderbook_path = "data/orderbook/"

# import googlecloudprofiler
# if depth_table == "btcusdt_depth_0ms":
#     try:
#         googlecloudprofiler.start(
#             service='btcusdt_with_copy_njit_50_3',
#             # verbose is the logging level. 0-error, 1-warning, 2-info,
#             # 3-debug. It defaults to 0 (error) if not set.
#             verbose=3,
#         )
#     except (ValueError, NotImplementedError) as exc:
#         print(exc)  # Handle errors here

# Conenct to Binance API
client = load_client()

DATABASE_DETAILS = [DB_HOST, DB_PASSWORD, DB_USER, DB_NAME]

# Connect to database
cur, engine, conn = connect_to_database(
    DATABASE_DETAILS[0], DATABASE_DETAILS[1], DATABASE_DETAILS[2], DATABASE_DETAILS[3]
)


# This code speeds up the sql insert somehow
@event.listens_for(engine, "before_cursor_execute")
def receive_before_cursor_execute(
    conn, cursor, statement, params, context, executemany
):
    if executemany:
        cursor.fast_executemany = True
        cursor.commit()


depth_level = 20  # Default 100; max 1000. Valid limits:[5, 10, 20, 50, 100, 500, 1000]
# depth_table = "btcusdt_depth_0ms"  ## TESTING OVERIDE
depth_pair = depth_table.split("_")[0]
# depth_pair = "btcusdt"  ## TESTING OVERIDE

try:
    depth_snapshot
except NameError:
    depth_snapshot = None

exchange_info = client.get_exchange_info()

if depth_snapshot is None:
    depth_snapshot = client.get_order_book(
        market="futures", symbol=depth_pair.upper(), limit=depth_level
    )

# Drop any event where u is <= lastUpdateId in the snapshot. so keep any >
rows = []
while len(rows) == 0:
    cur.execute(
        "SELECT * FROM {0} WHERE l > {1} ORDER BY l;".format(
            depth_table, depth_snapshot["lastUpdateId"]
        )
    )
    rows = cur.fetchall()
    time.sleep(1)


old_existing_bids = None
old_existing_asks = None
old_mid_price = None

existing_bids = [list(map(float, i)) for i in depth_snapshot["bids"]]
existing_asks = [list(map(float, i)) for i in depth_snapshot["asks"]]
while True:
    proccessed_row_list = []
    processed_bids_list = np.zeros((len(rows), depth_level, 2))
    processed_bids_list[:] = np.nan
    processed_asks_list = np.zeros((len(rows), depth_level, 2))
    processed_asks_list[:] = np.nan
    current_time = None
    print("rows" + str(len(rows)))
    existing_bids = np.asarray(existing_bids)
    existing_asks = np.asarray(existing_asks)
    different_count = 0
    for i in range(len(rows)):  # For one new row of bids
        prices_bids = rows[i][5][::2]
        volumes_bids = rows[i][5][1::2]
        prices_asks = rows[i][6][::2]
        volumes_asks = rows[i][6][1::2]
        book_type = "bids"

        if len(prices_bids) > 0:
            nan_index = None
            nan_index_bids = get_nans_index_from_2d_array_numba(existing_bids)
            if nan_index_bids is not None:
                existing_bids = np.delete(existing_bids, nan_index_bids, axis=0)

            existing_bids, nan_index = process_new_bid_or_ask_manager(
                existing_bids, prices_bids, volumes_bids, book_type
            )
            if nan_index is not None:

                existing_bids = np.delete(existing_bids, nan_index, axis=0)
            if book_type == "bids":
                existing_bids = existing_bids * -1
                existing_bids = existing_bids[existing_bids[:, 0].argsort()]
                existing_bids = existing_bids * -1

        book_type = "asks"
        if len(prices_asks) > 0:
            nan_index_asks = get_nans_index_from_2d_array_numba(existing_asks)
            if nan_index_asks is not None:
                existing_asks = np.delete(existing_asks, nan_index_asks, axis=0)
            existing_asks, nan_index = process_new_bid_or_ask_manager(
                existing_asks, prices_asks, volumes_asks, book_type
            )
            if nan_index is not None:
                existing_asks = np.delete(existing_asks, nan_index, axis=0)
            if book_type == "asks":

                existing_asks = existing_asks[existing_asks[:, 0].argsort()]

        mid_price = (existing_asks[0][0] + existing_bids[0][0]) / 2
        spread = existing_asks[0][0] - existing_bids[0][0]
        imbalance = existing_bids[0][1] / (existing_bids[0][1] - existing_asks[0][1])
        microprice = mid_price + spread * (imbalance - 0.5)
        best_bid = existing_bids[0][0]
        best_ask = existing_asks[0][0]

        write = False
        if i > 0:
            if old_best_bid != best_bid or old_best_ask != best_ask:
                write = True
                different_count = different_count + 1

        old_best_bid = best_bid
        old_best_ask = best_ask

        existing_bids = trim_existing_bids_or_asks(existing_bids, depth_level)
        existing_asks = trim_existing_bids_or_asks(existing_asks, depth_level)
        if write == True:

            processed_bids_list[i] = existing_bids[
                0:depth_level
            ]  # takes up alot of time
            processed_asks_list[i] = existing_asks[
                0:depth_level
            ]  # takes up alot of time

            proccessed_row_list.append(
                {
                    "event_time": rows[i][0],
                    "transaction_time": rows[i][1],
                    "first_update_id": rows[i][2],
                    "last_update_id": rows[i][3],
                    "last_update_id_previous": rows[i][4],
                    "mid_price": mid_price,
                    "spread": spread,
                    "imbalance": imbalance,
                    "microprice": microprice,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                }
            )

    print("different_count" + str(different_count))

    new_rows = False
    while new_rows == False:
        cur.execute(
            "SELECT * FROM {0} WHERE l > {1} ORDER BY l;".format(
                depth_table, rows[-1][3]
            )
        )
        next_rows = cur.fetchall()
        if len(next_rows) > 0:
            new_rows = True
            rows = next_rows
        else:
            time.sleep(0.5)

    df = pd.DataFrame(proccessed_row_list)
    print("df" + str(len(df)))
    table_name = depth_pair + "_orderbook"
    print(table_name)

    # Put the first row in with to_sql so that the table exists
    df.to_sql(
        table_name, engine, if_exists="append", method=psql_insert_copy, index=False
    )

    # can change the 1000's etc for desired orderbook length

    print("processed_bids_list" + str(len(processed_bids_list)))
    print("processed_asks_list" + str(len(processed_asks_list)))

    hdf5_epath = orderbook_path + table_name + ".hdf5"

    if os.path.exists(orderbook_path + table_name + ".hdf5") == False:
        h5f = tb.open_file(hdf5_epath, mode="a")
        dataGroup = h5f.create_group(h5f.root, "MyData")

        print("processed_bids_list.shape" + (str(processed_bids_list.shape)))
        print("processed_asks_list.shape" + (str(processed_asks_list.shape)))

        bids = h5f.create_earray(
            dataGroup, "bids", obj=processed_bids_list, shape=(0, depth_level, 2)
        )
        asks = h5f.create_earray(
            dataGroup, "asks", obj=processed_asks_list, shape=(0, depth_level, 2)
        )

        print("bids" + str(len(bids)))
        print("asks" + str(len(asks)))

        h5f.close()
        bids = None
        asks = None
    else:
        h5f = tb.open_file(hdf5_epath, mode="r+")
        print("processed_bids_list.shape" + (str(processed_bids_list.shape)))
        print("processed_asks_list.shape" + (str(processed_asks_list.shape)))
        h5f.root.MyData.bids.append(processed_bids_list)
        h5f.root.MyData.asks.append(processed_asks_list)
        h5f.close()
        inserted = 1

## Will need to split up the files every n'th row and then
## label with first_row,last_row or first last_update id, last last_update id

# nohup python json_to_postgres.py > json_to_postgres.log &
# nohup python socket_api_array.py > socket_api_array.log &
# nohup bash start_orderbook_symbol_scripts.sh &
# python manage_local_orderbook.py --symbol btcusdt_depth_0ms
