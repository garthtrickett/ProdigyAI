# -*- coding: utf-8 -*-
print("script started")
import os
import os.path
import tables as tb

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
from json import loads

import sqlalchemy
from sqlalchemy import create_engine, event
from sqlalchemy.dialects import postgresql

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


def remove_duplicates(list_with_duplicates):
    final_list = []
    for num in list_with_duplicates:
        if num not in final_list:
            final_list.append(num)
    return final_list


r = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo")
exchange_info = loads(r.content.decode())
symbol_information_list = exchange_info["symbols"]
symbol_list = []
for symbol_information_dict in symbol_information_list:
    symbol_list.append(symbol_information_dict["symbol"].lower())


str1 = ",".join(symbol_list)

try:
    get_ipython()
    check_if_ipython = True
    path = "orderbook_symbol_string.txt"

except Exception as e:
    check_if_ipython = False

    split_cwd = cwd.split("/")
    last_string = split_cwd.pop(-1)
    cwd = cwd.replace(last_string, "")
    os.chdir(cwd)
    path = "binance_api/orderbook_symbol_string.txt"

with open(path, "w+") as text_file:
    text_file.write(str1)
