"""
Gathers all real-time data from Binance websocket API for a single symbol.
@author: Jeffrey Wardman (Modified by the Rat Prince)
Created on 23rd Jan, 2019.
Documentation in each function is from https://github.com/binance-exchange/binance-official-api-docs/blob/master/web-socket-streams.md.
TO DO:
"""

# make /tmp/csv and give it the chown -R postgres:postgres and chmod 777 if it doesnt already
# then run the scripts
import pickle
import argparse
import fnmatch
import logging
import os
import sys
import time
import json
import csv
import requests
from json import loads

sys.path.append("..")
from argparse import ArgumentParser
from datetime import datetime

import dateutil.relativedelta

import third_party_libraries.python_binance.binance
from third_party_libraries.python_binance.binance.client import Client
from third_party_libraries.python_binance.binance.enums import (
    KLINE_INTERVAL_1MINUTE,
    WEBSOCKET_DEPTH_20,
)
from third_party_libraries.python_binance.binance.websockets import BinanceSocketManager

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# from twisted.internet import reactor
from sqlalchemy import create_engine, event

from support_files.locations import DB_HOST, DB_NAME, DB_PASSWORD, DB_USER, SaveDir
from support_files.locations import binance_apiKey as binKey
from support_files.locations import binance_apiSecret as binSecret
from library.core import *

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

client = load_client()

print("client loaded")
market = "futures"
# Getting all the info about binance exchange


exchange_info = get_exchange_info()

# Keep the symbols have margin trading
symbol_list = get_symbol_list(exchange_info, market=market)

# This is for running multiple python processes via bash pairs 0:5, pairs 6:10 etc
if check_if_ipython is False and args.symbol_split is not None:
    symbol_list = symbol_list[int(symbol_subset_array[0]) : int(symbol_subset_array[1])]

# Overide symbol list
# selected_symbols = ["ETHBTC"]
# symbol_list = selected_symbols
# symbol_list = symbol_list[0:5]

interval = KLINE_INTERVAL_1MINUTE
depth = WEBSOCKET_DEPTH_20

made_tables_set = set()

pairs_dict = {}  # holds everything
individual_pair_dict = {}  # holds individual symbols socket info

sockets = [
    "@aggTrade",
    "@trade",  # @trade for spot/margin tradin (@trade isnt on the api docs for futures and appears to be realtime)
    "@markPrice",  # this doesnt exist for spot/margin trading
    "@kline_" + interval,
    # "@depth" + depth + "@100ms",
    # "@depth@0ms",  # 100ms for spot/margin trading
    # "@bookTicker",
    # "@forceOrder",  # this doesnt exist for spot/margin trading
]

# building pairs_dict
for symbol in symbol_list:
    individual_pair_dict = {}
    for socket in sockets:
        trade_list = []
        individual_pair_dict.update({socket: trade_list})
    pairs_dict.update({symbol: individual_pair_dict})

path = "/tmp/csv/"

if not os.path.exists(path):
    os.makedirs(path)


def process_message(msg, symbol_list=symbol_list):
    path = "/tmp/csv/"
    for symbol in symbol_list:
        for socket in sockets:
            if msg["stream"] == symbol.lower() + socket:

                table_name = (symbol + socket).replace("@", "_").lower()
                if len(msg["data"]) > 0:
                    if msg["stream"] == symbol.lower() + "@kline_1m":
                        for row in [msg["data"]]:
                            pairs_dict[symbol.upper()][socket].append(row["k"])
                    else:
                        pairs_dict[symbol.upper()][socket].append(msg["data"])

                    if (
                        len(pairs_dict[symbol.upper()][socket]) > 1000
                    ):  # Number here is how many rows before saving to json
                        try:
                            # with open(path + table_name + '.json',
                            #           'w',
                            #           encoding='utf-8') as fout:
                            #     json.dump(pairs_dict[symbol.upper()][socket],
                            #               fout,
                            #               ensure_ascii=False,
                            #               indent=4)

                            pickle.dump(
                                pairs_dict[symbol.upper()][socket],
                                open(path + table_name + ".p", "wb"),
                            )

                            pairs_dict[symbol.upper()][
                                socket
                            ] = []  # wipe the dict after each sql insert
                            print(symbol + socket + "dict written")
                        except Exception as e:
                            print(e)
                            pass


def get_realtime_data(symbol_list, binKey, binSecret):
    symbol_list = [x.lower() for x in symbol_list]
    client = Client(binKey, binSecret)

    bm = BinanceSocketManager(client, market=market, user_timeout=1)

    stream_names = []
    for symbol in symbol_list:
        symbol_stream_names = []
        for socket in sockets:
            symbol_stream_names.append(symbol + socket)

        stream_names.extend(symbol_stream_names)
    conn_key = bm.start_multiplex_socket(stream_names, process_message)
    bm.start()

    # if condition: ##
    #     reactor.stop()
    #     bm.stop_socket(conn_key)  # may not stop when expected. use reactor instaead. added as a safety measure.


get_realtime_data(symbol_list, binKey, binSecret)
