print("script started")
import os
import argparse
import os.path
import tables as tb
import yaml
import wandb
import shutil

cwd = os.getcwd()
import sys
import h5py
import time
import math

sys.path.append("..")

import pathlib

from third_party_libraries.python_binance.binance.enums import (
    KLINE_INTERVAL_1MINUTE,
    WEBSOCKET_DEPTH_20,
)
from library.core import *
import pandas as pd
import numpy as np
import getpass
user = getpass.getuser()

from sqlalchemy import event
from numba import njit

from support_files.locations import (
    binance_apiKey as binKey,
    binance_apiSecret as binSecret,
    SaveDir,
    DB_HOST,
    DB_PASSWORD,
    DB_USER,
    DB_NAME,
)

orderbook_path = "data/orderbook/"
# Sorting out whether we are using the ipython kernel or not
try:
    get_ipython()
    check_if_ipython = True
    path_adjust = "../"
except Exception:
    check_if_ipython = False
    split_cwd = cwd.split("/")
    last_string = split_cwd.pop(-1)
    cwd = cwd.replace(last_string, "")
    os.chdir(cwd)
    parser = argparse.ArgumentParser(description="Binance Socket API")
    parser.add_argument("-s", "--symbol", type=str, help="symbol")
    parser.add_argument("-f",
                        "--is_finished",
                        type=str,
                        help="Is this a continuation of preempted instance?")
    parser.add_argument("-r",
                        "--resuming",
                        type=str,
                        help="Is this a continuation of preempted instance?")
    parser.add_argument("-u",
                        "--user",
                        type=str,
                        help="Stage of Preprocesssing")
    args = parser.parse_args()

    if args.symbol != None:
        depth_table = args.symbol

    path_adjust = ""

yaml_path = path_adjust + "yaml/orderbook_preprocessing.yaml"
with open(yaml_path) as file:
    yaml_dict = yaml.load(file, Loader=yaml.FullLoader)

config_dictionary = dict(yaml=yaml_path, params=yaml_dict)

wandb.init(
    dir="/home/" + user + "/ProdigyAI/",
    project="prodigyai",
    config=config_dictionary,
)

pair = wandb.config['params']['pair']['value']
source_type = wandb.config['params']['source_type']['value']

np.set_printoptions(
    precision=50)  # This makes sure numpy print outs aren't rounded

### GOOGLES GCLOUD PROFILER
# import googlecloudprofiler
# try:
#     googlecloudprofiler.start(
#         service='btcusdt_with_copy_njit',
#         # verbose is the logging level. 0-error, 1-warning, 2-info,
#         # 3-debug. It defaults to 0 (error) if not set.
#         verbose=3,
#     )
# except (ValueError, NotImplementedError) as exc:
#     print(exc)  # Handle errors here

very_start = time.time()

# Conenct to Binance API
DATABASE_DETAILS = [DB_HOST, DB_PASSWORD, DB_USER, DB_NAME]

# Connect to database
cur, engine, conn = connect_to_database(DATABASE_DETAILS[0],
                                        DATABASE_DETAILS[1],
                                        DATABASE_DETAILS[2],
                                        DATABASE_DETAILS[3])


# This code speeds up the sql insert somehow
@event.listens_for(engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, params, context,
                                  executemany):
    if executemany:
        cursor.fast_executemany = True
        cursor.commit()


### Start loading DB and .h5 Files ###
# Take just the first # rows for testing if 0 take all
limit = 0
print("loading db start")
start = time.time()

if limit == 0:
    cur.execute("SELECT * FROM " + pair + "_orderbook ORDER BY last_update_id")
else:
    cur.execute("SELECT * FROM " + pair +
                "_orderbook ORDER BY last_update_id LIMIT " + str(limit))

rows_orderbook = cur.fetchall()
end = time.time()
print("loading db finished" + str(start - end))
original_file_path = path_adjust + orderbook_path + pair + ".hdf5"
path = path_adjust + orderbook_path + pair + "_copy.hdf5"

# Monday, 6 April 2020 08:09:16.282 to Friday, 10 April 2020 11:03:59.352 4 days
# 8335610 / 4 = a lil more than 2 million rows a day

file = pathlib.Path(path)
if file.exists():
    print("File exist")
else:
    print("File not exist so copy the original to this path")
    dest = shutil.copyfile(original_file_path, path)

print("loading .5df5 files start")
start = time.time()
h5f = tb.open_file(path, mode="r+")
if limit == 0:
    bids = h5f.root.MyData.bids[:]
    asks = h5f.root.MyData.asks[:]
else:
    bids = h5f.root.MyData.bids[:limit + 100]
    asks = h5f.root.MyData.asks[:limit + 100]

h5f.close()

end = time.time()
print("loading .5df5 files end" + str(start - end))

bids[10][0][0] == rows_orderbook[10][-2]
asks[10][0][0] == rows_orderbook[10][-1]
if len(asks) < len(rows_orderbook):
    print("case1")
    rows_orderbook = rows_orderbook[:len(asks)]
elif len(asks) >= len(rows_orderbook):
    print("case2")
    asks = asks[:len(rows_orderbook)]
    bids = bids[:len(rows_orderbook)]

### Finish loading DB and .h5 Files ###

### If this is true each row will use the ID
if source_type == "orderbook_pinned_by_trades":
    ## ORDERBOOKS PINNED TO TRADES
    cur.execute('SELECT * FROM ' + pair +
                '_trade WHERE "T" >= {0} AND "T" <= {1} ORDER BY "T";'.format(
                    rows_orderbook[0][1], rows_orderbook[-1][1]))
    rows_trade = cur.fetchall()

    rows_trade = np.asarray(rows_trade)
    rows_trade = rows_trade[:, [1, 2, 4, 5, 6]].astype(float)
    rows_orderbook = np.asarray(rows_orderbook)
    rows_orderbook[0]
    rows_trade[0]

    duplicate_volume = 0
    duplicate_counter = 0
    rows_trade_duplicates_averaged = np.zeros(rows_trade.shape)
    rows_trade_duplicates_averaged[:] = np.nan
    for i in range(len(rows_trade)):
        if i > 0:
            if rows_trade[i - 1][1] == rows_trade[i][1]:
                duplicate_counter = duplicate_counter + 1
                if duplicate_volume == 0:
                    duplicate_volume = rows_trade[i - 1][4] + rows_trade[i][4]
                    duplicate_price = (
                        rows_trade[i - 1][3] * rows_trade[i - 1][4] +
                        rows_trade[i][3] * rows_trade[i][4])

                else:
                    duplicate_volume = duplicate_volume + rows_trade[i][4]
                    duplicate_price = duplicate_price + rows_trade[i][
                        3] * rows_trade[i][4]

            else:
                rows_trade_duplicates_averaged[i] = rows_trade[i]

            if duplicate_volume != 0 and rows_trade[i -
                                                    1][1] != rows_trade[i][1]:

                rows_trade_duplicates_averaged[i] = rows_trade[i]
                rows_trade_duplicates_averaged[i][4] = duplicate_volume
                rows_trade_duplicates_averaged[i][
                    4] = duplicate_price / duplicate_volume
                duplicate_volume = 0
                duplicate_price = 0
                duplicate_counter = 0
        else:
            rows_trade_duplicates_averaged[i] = rows_trade[i]

    rows_trade_duplicates_averaged = rows_trade_duplicates_averaged[
        ~np.isnan(rows_trade_duplicates_averaged).any(axis=1)]

    level = 10
    combined_trades_and_orderbook_array = combine_trades_and_orderbook(
        rows_trade_duplicates_averaged, rows_orderbook, bids, asks, level)
    len(rows_trade)

    df = pd.DataFrame(combined_trades_and_orderbook_array)
    df = df.rename(columns={0: "date_time", 3: "close"})
    df = df.set_index("date_time")
    df.index = pd.to_datetime(df.index, unit="ms")
    volumes = df.iloc[:, 4:]
    df = df[["close"]]

elif source_type == "only_orderbook":
    ### ONLY ORDERBOOK
    rows_orderbook_np_array = np.asarray(rows_orderbook)
    rows_orderbook_np_array = rows_orderbook_np_array[:, [1, 5]]
    df = pd.DataFrame(rows_orderbook_np_array)
    index_np_array = np.asarray(df.iloc[:, 0])
    orderbook_depth = 10

    averaged_bids, averaged_asks = average_bids_or_asks(
        bids, asks, index_np_array, orderbook_depth, 0, 0, len(index_np_array))

    reshaped_bids_prices = averaged_bids.reshape(
        len(averaged_bids),
        averaged_bids.shape[1] * averaged_bids.shape[2])[:, ::2]
    reshaped_bids_volumes = averaged_bids.reshape(
        len(averaged_bids),
        averaged_bids.shape[1] * averaged_bids.shape[2])[:, 1::2]
    reshaped_asks_prices = averaged_asks.reshape(
        len(averaged_asks),
        averaged_asks.shape[1] * averaged_asks.shape[2])[:, ::2]
    reshaped_asks_volumes = averaged_asks.reshape(
        len(averaged_asks),
        averaged_asks.shape[1] * averaged_asks.shape[2])[:, 1::2]

    best_bids = reshaped_bids_prices[:, 0]
    best_asks = reshaped_asks_prices[:, 0]

    mid_prices = np.zeros(len(best_bids))
    mid_prices[:] = np.nan

    for i in range(len(best_bids)):
        mid_prices[i] = (best_bids[i] + best_asks[i]) / 2

    reshaped_bids_and_asks = np.concatenate(
        (
            reshaped_bids_prices,
            reshaped_bids_volumes,
            reshaped_asks_prices,
            reshaped_asks_volumes,
        ),
        axis=1,
    )

    bids_and_asks_df = pd.DataFrame(reshaped_bids_and_asks)
    bids_and_asks_df.index = df.iloc[:, 0]
    bids_and_asks_df.index.name = "date_time"
    volumes = bids_and_asks_df.dropna()
    print(len(volumes))

    mid_prices_df = pd.DataFrame(mid_prices)
    mid_prices_df.index = df.iloc[:, 0]
    mid_prices_df.index.name = "date_time"

    df = mid_prices_df.dropna()

# Create the txt file string
parameter_string = wandb.run.id

# save data dataframe
df_index_as_epoch = np.asarray(df.index.astype(np.int64))
df_np_array = np.asarray(df)

volumes_index_as_epoch = np.asarray(volumes.index.astype(np.int64))
volumes_np_array = np.asarray(volumes)

h5f = h5py.File(
    path_adjust + "data/orderbook_preprocessed/" + parameter_string + "_" +
    pair + ".h5", "w")
h5f.create_dataset("volumes_index_as_epoch", data=volumes_index_as_epoch)
h5f.create_dataset("volumes_np_array", data=volumes_np_array)
h5f.create_dataset("df_index_as_epoch", data=df_index_as_epoch)
h5f.create_dataset("df_np_array", data=df_np_array)
h5f.close()

with open(path_adjust + "temp/orderbook_data_name.txt", "w+") as text_file:
    text_file.write(parameter_string)
very_end = time.time()
print("full_script_time" + str(very_end - very_start))

# ### ONLY TRADES

# df = pd.DataFrame(rows_trade)
# df = df.drop(columns=[0, 1, 3, 4, 7])
# df = df.rename(columns={2: "date_time", 5: "close", 6: "volume"})
# df = df.set_index("date_time")
# print("converting index to date_time")
# df.index = pd.to_datetime(df.index, unit="ms")

# print("index converted")
# # Should do something else than drop the duplicates (maybe trades doesnt have duplicate indexes rather than aggtrades)
# df = df.loc[~df.index.duplicated(keep="first")]
# df = df.head(10000)

# cols = df.columns[df.dtypes.eq("object")]
# df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
# len(df)
# # data = df

# rows_trade_np_array = np.asarray(rows_trade)
# rows_trade_np_array_boolean_columns = rows_trade_np_array[:, [7]]
# rows_trade_np_array_float_columns = rows_trade_np_array[:, [1, 2, 4, 5, 6]].astype(
#     float
# )

# # create a new numpy array from zeroes that has the length of
# # filtered rows and enough columns store orderbook and trade data
# for i in range(len(filtered_rows)):
#     for j in range(len(rows_trade_np_array)):
#         # Get the first orderbook row transaction time
#         # that is greater than the a trade transaction time
#         # ( COULD ALSO DO THIS THE OTHER WAY ROUND
#         # AND HAVE THE SAMPLES HAVE TRADE TIMES)
#         # IE what does the orderbook like just before a trade rather than
#         # what was the last trade right before and orderbook update
#         if rows_trade_np_array_float_columns[j][1] <= filtered_rows[i][1]:
#             import pdb

#             pdb.set_trace()
#             # Do something then break
#             break

### About isbuyermaker
# If isBuyerMaker is true for the trade, it means that the order
# of whoever was on the buy side, was sitting as a bid in the orderbook
# for some time (so that it was making the market) and then someone came in
# and matched it immediately (market taker). So, that specific trade will now
# qualify as SELL and in UI highlight as redish. On the opposite
# isBuyerMaker=false trade will qualify as BUY and highlight greenish.
