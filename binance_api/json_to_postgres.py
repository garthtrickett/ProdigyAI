"""
Gathers all real-time data from Binance websocket API for a single symbol.
@author: Jeffrey Wardman (Modified by the Rat Prince)
Created on 23rd Jan, 2019.
Documentation in each function is from https://github.com/binance-exchange/binance-official-api-docs/blob/master/web-socket-streams.md.
TO DO:
"""
import pickle
import argparse
import fnmatch
import logging
import os
import sys
import time
import json
import csv
import glob
import pwd
import grp
import os
import sqlalchemy
from sqlalchemy.dialects import postgresql

sys.path.append("..")
from argparse import ArgumentParser
from datetime import datetime

import dateutil.relativedelta

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# from twisted.internet import reactor
from sqlalchemy import create_engine, event

import third_party_libraries.python_binance.binance
from third_party_libraries.python_binance.binance.client import Client
from third_party_libraries.python_binance.binance.enums import (
    KLINE_INTERVAL_1MINUTE,
    WEBSOCKET_DEPTH_20,
)
from third_party_libraries.python_binance.binance.websockets import BinanceSocketManager

from support_files.locations import DB_HOST, DB_NAME, DB_PASSWORD, DB_USER, SaveDir
from support_files.locations import binance_apiKey as binKey
from support_files.locations import binance_apiSecret as binSecret
from library.core import *

print("finished loading scripts")

logging.basicConfig(level=logging.DEBUG)

try:
    get_ipython()
    check_if_ipython = True

except Exception as e:
    check_if_ipython = False
    parser = argparse.ArgumentParser(description="Binance Socket API")
    parser.add_argument("-s", "--symbol_split", type=str, help="symbol_split")
    args = parser.parse_args()

    if args.symbol_split != None:
        symbol_subset = args.symbol_split
        symbol_subset_array = symbol_subset.split(":")

# Postgres engine
cur, engine, conn = connect_to_database(DB_HOST, DB_PASSWORD, DB_USER, DB_NAME)


# This code speeds up the sql insert somehow
@event.listens_for(engine, "before_cursor_execute")
def receive_before_cursor_execute(
    conn, cursor, statement, params, context, executemany
):
    if executemany:
        cursor.fast_executemany = True
        cursor.commit()


path = "/tmp/csv/"

if not os.path.exists(path):
    os.makedirs(path)

os.chdir(path)

while True:
    for file in glob.glob("*.p"):
        table_name = file.split(".")[0]
        # print(table_name)
        myCmd = "sudo chown -R postgres:postgres " + path + file
        os.system(myCmd)
        try:
            p = pickle.load(open(file, "rb"))
            # if "forceOrder" in table_name or "forceorder" in table_name:
            #     df = pd.DataFrame(p["o"])
            # else:
            #     df = pd.DataFrame(p)
            df = pd.DataFrame(p)
            df.to_sql(
                table_name, engine, if_exists="append", method="multi", index=False
            )
            os.remove(file)
            print(file + " inserted and deleted")
        except Exception as e:
            print(e)
            print("something broke for table name: " + table_name)
            pass

    time.sleep(1)
