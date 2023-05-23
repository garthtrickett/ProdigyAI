import websocket
import requests
from json import loads
import os.path
import tables as tb

print("script started")
import os
import argparse
import os.path


cwd = os.getcwd()
import sys

sys.path.append("..")

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


# import googlecloudprofiler

# try:
#     googlecloudprofiler.start(
#         service='btcusdt_with_third_party_numba',
#         # verbose is the logging level. 0-error, 1-warning, 2-info,
#         # 3-debug. It defaults to 0 (error) if not set.
#         verbose=3,
#     )
# except (ValueError, NotImplementedError) as exc:
#     print(exc)  # Handle errors here


class Client:
    def __init__(self, orderbook_path, depth_table):
        # local data management
        self.orderbook_path = orderbook_path
        self.depth = 1000
        self.stored_depth = 20
        self.symbol = depth_table
        self.socket = "@depth@0ms"
        self.orderbook = {}
        self.updates = 0
        self.last_best_bid = 0
        self.last_best_ask = 0
        self.write_count = 0
        self.count = 0
        self.new_snapshot_taken = 0
        self.new_snapshot = []
        self.write_time = time.time()
        self.on_close_count = 0
        self.snapshot_poll_count = 0
        self.stop_for_testing = 0

    def initiate(self):
        # create websocket connection
        self.ws = websocket.WebSocketApp(
            url="wss://fstream.binance.com/stream?streams=" + self.symbol + self.socket,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open,
        )

        self.ws.run_forever()

    # convert message to dict, process update
    def on_message(self, message):
        stream = loads(message)
        data = stream["data"]
        # if self.orderbook == {}:
        #     import pdb
        #     pdb.set_trace()

        ### Test if a later snapshot lines up with the local orderbook
        # print(self.count)

        # if self.count == 10000:
        #     self.new_snapshot_taken = 1
        #     self.new_snapshot = self.get_snapshot()
        #     self.new_snapshot['lastUpdateId']
        #     old_bids = self.orderbook['bids'][0:20]
        #     new_bids = np.asarray(
        #         self.new_snapshot['bids'][0:20]).astype(float)
        #     old_asks = self.orderbook['asks'][0:20]
        #     new_asks = np.asarray(
        #         self.new_snapshot['asks'][0:20]).astype(float)

        # if self.new_snapshot_taken == 1:
        #     print(self.new_snapshot['lastUpdateId'])
        #     print(self.orderbook['lastUpdateId'])
        #     if self.new_snapshot['lastUpdateId'] == self.orderbook[
        #             'lastUpdateId']:

        #         # self.orderbook['bids'][0:20] ==  np.asarray(self.new_snapshot['bids'][0:20]).astype(float)
        #         # self.orderbook['asks'][0:20] == np.asarray(self.new_snapshot['asks'][0:20]).astype(float)
        #         import pdb
        #         pdb.set_trace()

        # ### TESTING CODE
        # wait = 0
        # print(self.count)
        # if self.count == 1000:
        #     self.on_close_count = 1
        #     self.orderbook = {}
        #     wait = 1

        # check for orderbook, if empty retrieve
        if len(self.orderbook) == 0:
            time.sleep(0.1)
            snapshot = self.get_snapshot()
            self.snapshot_poll_count = self.snapshot_poll_count = (
                self.snapshot_poll_count + 1
            )

            ### TESTING CODE
            # if wait == 1:
            #     data["U"] = snapshot["lastUpdateId"] + 100

            if data["U"] <= snapshot["lastUpdateId"]:
                self.orderbook = snapshot
            elif self.on_close_count > 0:
                while (
                    data["U"]
                    > snapshot[
                        "lastUpdateId"
                    ]  # or data["u"] < snapshot["lastUpdateId"]
                ):
                    time.sleep(0.1)
                    snapshot = self.get_snapshot()
                    print("snapshot_count" + str(self.snapshot_poll_count))
                    self.snapshot_poll_count = self.snapshot_poll_count = (
                        self.snapshot_poll_count + 1
                    )
                self.orderbook = snapshot

        # get lastUpdateId
        lastUpdateId = self.orderbook["lastUpdateId"]

        # drop any updates older than the snapshot
        if self.updates == 0:
            if data["U"] <= lastUpdateId and data["u"] >= lastUpdateId:
                self.orderbook["lastUpdateId"] = data["u"]
                self.orderbook["bids"] = np.asarray(self.orderbook["bids"]).astype(
                    float
                )
                self.orderbook["asks"] = np.asarray(self.orderbook["asks"]).astype(
                    float
                )
                self.process_updates(data)
                # self.write_orderbook_bids_and_asks_to_pytables()
                # df_dict = self.create_dict_from_orderbook()
                # df = pd.DataFrame([df_dict])
                # table_name = self.symbol + "_orderbook"
                # df.to_sql(
                #     table_name,
                #     engine,
                #     if_exists="append",
                #     method=psql_insert_copy,
                #     index=False,
                # )
                self.updates = 1
                self.last_best_bid = self.orderbook["bids"][0][0]
                self.last_best_ask = self.orderbook["asks"][0][0]
            else:
                print("lastupdateid" + str(lastUpdateId))
                print("dataU" + str(data["U"]))
                print("datau" + str(data["u"]))
                print("discard update")

        ### TESTING CODE
        # elif(True and self.stop_for_testing == 0):
        #     print("Out of sync, abort")
        #     self.stop_for_testing = 1
        #     self.on_close()

        # check if update still in sync with orderbook
        elif data["pu"] == self.previous_events_u:
            self.orderbook["lastUpdateId"] = data["u"]
            self.orderbook["firstUpdateId"] = data["U"]
            self.orderbook["lastUpdateIdPrevious"] = data["pu"]
            self.orderbook["E"] = data["E"]
            self.orderbook["T"] = data["T"]

            self.orderbook["bids"] = np.asarray(self.orderbook["bids"]).astype(float)
            self.orderbook["asks"] = np.asarray(self.orderbook["asks"]).astype(float)
            self.process_updates(data)

            if (
                self.last_best_bid != self.orderbook["bids"][0][0]
                or self.last_best_ask != self.orderbook["asks"][0][0]
            ):

                self.write_orderbook_bids_and_asks_to_pytables()

                df_dict = self.create_dict_from_orderbook()
                df = pd.DataFrame([df_dict])
                table_name = self.symbol + "_orderbook"
                df.to_sql(
                    table_name,
                    engine,
                    if_exists="append",
                    method=psql_insert_copy,
                    index=False,
                )

                self.write_count = self.write_count + 1

                # if self.write_count > 20:
                #     self.write_count = 0
                #     self.on_close()
                if time.time() - self.write_time > 60:
                    print(self.write_count)
                    self.write_time = time.time()
            self.last_best_bid = self.orderbook["bids"][0][0]
            self.last_best_ask = self.orderbook["asks"][0][0]
        else:
            print("Out of sync, abort")
            self.on_close()

        self.previous_events_u = data["u"]
        self.count = self.count + 1

    # catch errors
    def on_error(self, error):
        print(error)

    # run when websocket is closed
    def on_close(self):
        print("### closed ###")
        self.on_close_count = self.on_close_count + 1
        self.orderbook = {}
        self.updates = 0
        self.last_best_bid = 0
        self.last_best_ask = 0
        self.initiate()

    # run when websocket is initialised
    def on_open(self):
        print("Connected to Binance\n")

    # Loop through all bid and ask updates, call manage_orderbook accordingly
    def process_updates(self, data):
        for update in data["b"]:
            update = np.asarray(update).astype(float)
            self.manage_orderbook("bids", update)
        for update in data["a"]:
            update = np.asarray(update).astype(float)
            self.manage_orderbook("asks", update)

    def manage_orderbook(self, side, update):

        price = update[0]
        qty = update[1]
        self.orderbook[side] = self.orderbook[side].astype(float)

        ammended, nan_exists, self.orderbook[side] = ammend_new_prices(
            self.orderbook[side], update
        )

        if nan_exists == 1:
            nan_indexes = get_nans_index_from_2d_array_numba(
                self.orderbook[side].astype(float)
            )
            if nan_indexes is not None:
                self.orderbook[side] = np.delete(
                    self.orderbook[side], nan_indexes, axis=0
                )

        if side == "bids":
            side_int = 0
        elif side == "asks":
            side_int = 1

        if ammended == 0:
            if qty != 0:
                self.orderbook[side] = append_new_prices(
                    self.orderbook[side], update, side_int
                )

        # cut the bids and asks where bid_depth  > self.depth
        self.orderbook[side] = self.orderbook[side][0 : self.depth]

    # retrieve orderbook snapshot
    def get_snapshot(self):
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/depth?symbol="
            + self.symbol.upper()
            + "&limit="
            + str(self.depth)
        )
        return loads(r.content.decode())

    def write_orderbook_bids_and_asks_to_pytables(self):
        bids = np.asarray(self.orderbook["bids"]).astype(float)[0 : self.stored_depth]
        asks = np.asarray(self.orderbook["asks"]).astype(float)[0 : self.stored_depth]
        bids = trim_existing_bids_or_asks(bids, self.stored_depth)
        asks = trim_existing_bids_or_asks(asks, self.stored_depth)
        bids = bids.reshape(1, bids.shape[0], bids.shape[1])
        asks = asks.reshape(1, asks.shape[0], asks.shape[1])
        hdf5_epath = orderbook_path + self.symbol + ".hdf5"
        if os.path.exists(hdf5_epath) == False:
            h5f = tb.open_file(hdf5_epath, mode="a")
            dataGroup = h5f.create_group(h5f.root, "MyData")

            h5f.create_earray(dataGroup, "bids", obj=bids)
            h5f.create_earray(dataGroup, "asks", obj=asks)

            h5f.close()
            bids = None
            asks = None
        else:
            h5f = tb.open_file(hdf5_epath, mode="r+")
            h5f.root.MyData.bids.append(bids)

            h5f.root.MyData.asks.append(asks)

            h5f.close()

    def create_dict_from_orderbook(self):
        mid_price = (self.orderbook["bids"][0][0] + self.orderbook["asks"][0][0]) / 2
        spread = self.orderbook["asks"][0][0] - self.orderbook["bids"][0][0]

        imbalance = self.orderbook["bids"][0][1] / (
            self.orderbook["bids"][0][1] - self.orderbook["asks"][0][1]
        )
        microprice = mid_price + spread * (imbalance - 0.5)

        best_bid = self.orderbook["bids"][0][0]
        best_ask = self.orderbook["asks"][0][0]

        df_dict = {
            "event_time": self.orderbook["E"],
            "transaction_time": self.orderbook["T"],
            "first_update_id": self.orderbook["firstUpdateId"],
            "last_update_id": self.orderbook["lastUpdateId"],
            "last_update_id_previous": self.orderbook["lastUpdateIdPrevious"],
            "mid_price": mid_price,
            "spread": spread,
            "imbalance": imbalance,
            "microprice": microprice,
            "best_bid": best_bid,
            "best_ask": best_ask,
        }

        return df_dict

    # make new function that puts stuff in the db and hd5f


@njit
def ammend_new_prices(self_orderbook_side, update):
    ammended = 0
    nan_exists = 0
    price = update[0]
    qty = update[1]
    for x in range(0, len(self_orderbook_side)):
        if price == self_orderbook_side[x][0]:
            # when qty is 0 remove from orderbook, else
            # update values
            if qty == 0:
                self_orderbook_side[x] = [np.nan, np.nan]
                ammended = 1
                nan_exists = 1

                break
            else:
                self_orderbook_side[x] = np.asarray(update)
                ammended = 1
                break
    return ammended, nan_exists, self_orderbook_side


#  self.orderbook[side] = append_new_prices(self.orderbook[side], update, side_int)


@njit
def append_new_prices(self_orderbook_side, update, side_int):
    self_orderbook_side = np.append(self_orderbook_side, np.asarray(update))

    self_orderbook_side = self_orderbook_side.reshape(
        (int(len(self_orderbook_side) / 2), 2)
    )

    if side_int == 0:
        self_orderbook_side = self_orderbook_side * -1
        self_orderbook_side = self_orderbook_side[self_orderbook_side[:, 0].argsort()]
        self_orderbook_side = self_orderbook_side * -1

    elif side_int == 1:
        self_orderbook_side = self_orderbook_side[self_orderbook_side[:, 0].argsort()]

    return self_orderbook_side


if __name__ == "__main__":
    # create webscocket client
    client = Client(orderbook_path, depth_table)

    # run forever
    client.initiate()

# nohup python json_to_postgres.py > json_to_postgres.log &
# nohup python socket_api_array.py > socket_api_array.log &
# nohup bash start_orderbook_symbol_scripts.sh &
# nohup python orderbook_thirdparty_code.py > orderbook_thirdparty_code_btcusdt.log  --symbol btcusdt &
# python orderbook_thirdparty_code.py --symbol btcusdt
