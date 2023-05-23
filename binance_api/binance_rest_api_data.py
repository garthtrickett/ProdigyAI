import pandas as pd
from third_party_libraries.python_binance.client import Client
from third_party_libraries.python_binance.websockets import BinanceSocketManager
from third_party_libraries.python_binance.enums import (
    WEBSOCKET_DEPTH_20,
    KLINE_INTERVAL_1MINUTE,
)
import os

sys.path.append("..")
import dateparser
import datetime
from datetime import datetime
import tensorflow as tf
import pprint
import time
from library.core import connect_to_database

# from twisted.internet import reactor

from support_files.locations import (
    binance_apiKey as binKey,
    binance_apiSecret as binSecret,
    SaveDir,
    DB_HOST,
    DB_PASSWORD,
    DB_USER,
    DB_NAME,
)


def write_iterator_to_tf_record(aggtrades):
    """#todo: document"""
    day = 1
    new_file = 1
    for trade in agg_trades:
        if new_file == 1:
            date = time.strftime("%Y_%m_%d", time.localtime(trade["T"] / 1000))
            record_filename = symbol + "_" + str(date) + ".tfrecord"

            writer = tf.io.TFRecordWriter(symbol_data_path + "/" + record_filename)
            new_file = 0  # new file

        example = trade_example(trade)
        writer.write(example.SerializeToString())

        if trade["T"] > epoch_start_date + (day * day_in_milliseconds):
            writer.close()
            new_file = 1
            day = day + 1

    writer.close()


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def trade_example(trade):
    """#todo: document"""
    feature = {
        "Aggregate tradeId": _int64_feature(int(trade["a"])),
        "Price": _float_feature(float(trade["p"])),
        "Quantity": _float_feature(float(trade["q"])),
        "First tradeId": _int64_feature(int(trade["f"])),
        "Last tradeId": _int64_feature(int(trade["l"])),
        "Timestamp": _int64_feature(int(trade["T"])),
        "Was Buyer Maker": _int64_feature(trade["m"]),
        "Was Trade Best Price": _int64_feature(trade["M"]),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def load_client():
    """#todo: document"""
    # Got the api key and api secret to authenticate to the binance api
    from support_files.locations import (
        binance_apiKey as binKey,
        binance_apiSecret as binSecret,
        SaveDir,
    )

    # connecting to binance API
    client = Client(binKey, binSecret)

    return client


def get_first_and_last_row_of_table(engine, table_name, cur, order_by):
    """#todo: document"""
    try:
        # Get the last row in the database
        cur.execute(
            "SELECT * FROM {0} ORDER BY {0} desc limit 1;".format(table_name, order_by)
        )

        last_row = cur.fetchall()
        if order_by == "timestamp":
            last_row = last_row[0][8]
        elif order_by == "open_time":
            last_row = last_row[0][7]

    except Exception as e:
        print("error 2", e)

    try:
        # Get the first row in the database
        cur.execute(
            "SELECT * FROM {0} ORDER BY {0} asc limit 1;".format(table_name, order_by)
        )

        first_row = cur.fetchall()

        if order_by == "timestamp":
            first_row = first_row[0][8]
        elif order_by == "open_time":
            first_row = first_row[0][7]

    except Exception as e:
        print("error 2", e)

    return first_row, last_row


def get_symbol_list(exchange_info, selected_symbols):
    """#todo: document"""
    symbols_from_binance = exchange_info["symbols"]

    # Making a new list of symbols from the binance exchange information
    symbol_list = []
    for symbol in symbols_from_binance:
        symbol_list.append(symbol["symbol"])

    for symbol in selected_symbols:
        if symbol in symbol_list:
            print(symbol)
        else:
            selected_symbols.remove(symbol)

    return selected_symbols


# Get trade data
def get_rest_api_trade_data(start_date, end_date, selected_symbols, database_details):
    """#todo: document"""
    client = load_client()
    # Available candlesticks
    # 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w

    # Getting all the info about binance exchange
    exchange_info = client.get_exchange_info()

    cur, engine, conn = connect_to_database(
        database_details[0],
        database_details[1],
        database_details[2],
        database_details[3],
    )

    # Keep the symbols that are in the binance list
    list_of_symbols = get_symbol_list(exchange_info, selected_symbols)

    for symbol in list_of_symbols:
        table_name = (symbol + "_agg_trades").lower()

        epoch_start_date = int(
            datetime.timestamp(datetime.strptime(start_date, "%d %b, %Y")) * 1000
        )

        # path_format_start_date = time.strftime(
        #     '%Y_%m_%d', time.localtime(epoch_start_date / 1000))

        epoch_end_date = int(
            datetime.timestamp(datetime.strptime(end_date, "%d %b, %Y")) * 1000
        )

        # path_format_end_date = time.strftime(
        #     '%Y_%m_%d', time.localtime(epoch_end_date / 1000))

        # symbol_data_path = 'data/trade_rest_data/' + symbol

        # if not os.path.exists(symbol_data_path):
        #     os.mkdir(symbol_data_path)
        #     print("Directory ", symbol_data_path, " Created ")

        # while epoch_start_date < epoch_end_date:
        #     path_format_start_date = time.strftime(
        #         '%Y_%m_%d', time.localtime(epoch_start_date / 1000))

        #     # ETHBTC_2019_05_29.tfrecord

        #     symbol_data_date_path = symbol_data_path + '/' + symbol + '_' + path_format_start_date + '.tfrecord'
        #     exists = os.path.isfile(symbol_data_date_path)
        #     if exists:
        #         epoch_start_date = epoch_start_date + day_in_milliseconds
        #     else:
        #         break

        #  Start date string in UTC format or timestamp in milliseconds

        # # Writing data to tfrecord with tf.data api
        # def generator():
        #     for trade in agg_trades:
        #         example = trade_example(trade)
        #         yield example.SerializeToString()

        # serialized_features_dataset = tf.data.Dataset.from_generator(
        #     generator, output_types=tf.string, output_shapes=())

        # filename = 'test_tfdata.tfrecord'
        # writer = tf.data.experimental.TFRecordWriter(filename)
        # writer.write(serialized_features_dataset)

        # end = time.time()
        # print("tf data time=", (end - start))

        # Writing data to tfrecord with python api

        order_by = "timestamp"

        agg_trades = client.aggregate_trade_iter(
            symbol=symbol, start_str=epoch_start_date
        )

        if "earliest_row" in locals():
            del earliest_row
        if "latest_row" in locals():
            del latest_row

        if engine.dialect.has_table(engine, table_name):
            earliest_row, latest_row = get_first_and_last_row_of_table(
                engine, table_name, cur, order_by
            )

        # run the code x times
        count = 0
        count_two = 0
        current_trade_time = 0
        current_trade_id = 0
        agg_trades_list = []
        while current_trade_time < epoch_end_date:
            if current_trade_time > 0 and count_two > 600000:
                start_time = current_trade_time
                agg_trades = client.aggregate_trade_iter(
                    symbol=symbol, last_id=current_trade_id
                )
                count_two = 0

            for agg_trade in agg_trades:
                count = count + 1
                count_two = count_two + 1

                if agg_trade["T"] > epoch_end_date:
                    current_trade_time = agg_trade["T"]
                    current_trade_id = agg_trade["a"]
                    break

                if (
                    "earliest_row" not in locals()
                    or agg_trade["T"] < earliest_row
                    or agg_trade["T"] > latest_row
                ):

                    print(agg_trade)
                    agg_trade_dict = {}
                    agg_trade_dict["aggregate_trade_id"] = agg_trade["a"]
                    agg_trade_dict["price"] = agg_trade["p"]
                    agg_trade_dict["quantity"] = agg_trade["q"]
                    agg_trade_dict["first_trade_id"] = agg_trade["f"]
                    agg_trade_dict["last_trade_id"] = agg_trade["l"]
                    agg_trade_dict["timestamp"] = agg_trade["T"]
                    agg_trade_dict["is_buyer_market_maker"] = agg_trade["m"]
                    agg_trade_dict["ignore"] = agg_trade["M"]
                    agg_trades_list.append(agg_trade_dict)

                if count > 1000:
                    current_trade_time = agg_trade["T"]
                    current_trade_id = agg_trade["a"]
                    # Convert list of dicts into pandas dataframe
                    df = pd.DataFrame(agg_trades_list)

                    df = df.set_index("aggregate_trade_id")

                    # Put the dataframe into postgres sql table
                    try:
                        df.to_sql(table_name, engine, if_exists="append")
                    except Exception as e:
                        print(e)
                    agg_trades_list = []
                    count = 0

        # write_iterator_to_tf_record(aggtrades)

    # fatest way to store + fastest way to read into tensorflow

    # Three possible options for bulk loading into sql
    # https://www.mydatahack.com/how-to-bulk-load-data-into-postgresql-with-python/
    # https://www.codementor.io/bruce3557/graceful-data-ingestion-with-sqlalchemy-and-pandas-pft7ddcy6 with  https://docs.sqlalchemy.org/en/13/orm/tutorial.html
    # pg_bulkload


def get_restapi_klines(
    start_date, end_date, frequency, selected_symbols, database_details
):
    """#todo: document"""
    client = load_client()
    # Available candlesticks
    # 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w

    # Getting all the info about binance exchange
    exchange_info = client.get_exchange_info()

    # Keep the symbols that are in the binance list
    list_of_symbols = get_symbol_list(exchange_info, selected_symbols)

    cur, engine, conn = connect_to_database(
        database_details[0],
        database_details[1],
        database_details[2],
        database_details[3],
    )
    # For each of the symbols get the historical candlestick data and convert it into a pandas datframe
    # Put the pandas dataframes into the postgre sql file
    kline_dict = {}
    add_rows = 1
    for symbol in list_of_symbols:
        interval = frequency
        table_name = (symbol + "_kline_" + interval).lower()
        print(table_name)
        kline_list = []

        order_by = "open_time"
        ## Check if theres already some candles in the db
        if engine.dialect.has_table(engine, table_name):
            del first_row
            del last_row
            first_row, last_row = get_first_and_last_row_of_table(
                engine, table_name, cur, order_by
            )

            epoch_start_date = int(
                datetime.timestamp(datetime.strptime(start_date, "%d %b, %Y")) * 1000
            )
            epoch_end_date = int(
                datetime.timestamp(datetime.strptime(end_date, "%d %b, %Y")) * 1000
            )

            if epoch_start_date < first_row:
                if epoch_end_date < first_row:
                    # Get the earlier bit only
                    klines = client.get_historical_klines(
                        symbol, interval, start_date, end_date
                    )
                else:
                    klines = client.get_historical_klines(
                        symbol, interval, start_date, first_row
                    )
                    if last_row > epoch_end_date:
                        # Already got the later rows
                        pass
                    else:
                        klines_later = client.get_historical_klines(
                            symbol, interval, last_row, end_date
                        )
                        klines = klines + klines_later
            else:
                if epoch_start_date < last_row and epoch_end_date > last_row:
                    klines = client.get_historical_klines(
                        symbol, interval, last_row, epoch_end_date
                    )
                elif epoch_start_date == first_row and epoch_end_date == last_row:
                    add_rows = 0
                else:
                    klines = client.get_historical_klines(
                        symbol, interval, epoch_start_date, epoch_end_date
                    )
        else:
            interval = KLINE_INTERVAL_1MINUTE
            klines = client.get_historical_klines(
                symbol, interval, start_date, end_date
            )

        if add_rows != 0:
            kline_dict[symbol] = klines
            for kline in klines:
                kline_dict = {}
                kline_dict["open_time"] = kline[0]
                kline_dict["open"] = kline[1]
                kline_dict["high"] = kline[2]
                kline_dict["low"] = kline[3]
                kline_dict["close"] = kline[4]
                kline_dict["volume"] = kline[5]
                kline_dict["close_time"] = kline[6]
                kline_dict["quote_asset_volume"] = kline[7]
                kline_dict["number_of_trades"] = kline[8]
                kline_dict["taker_buy_base_asset_volume"] = kline[9]
                kline_dict["taker_buy_quote_asset_volume"] = kline[10]
                kline_list.append(kline_dict)

            # Convert list of dicts into pandas dataframe
            df = pd.DataFrame(kline_list)

            # set the index of the dataframe
            # df = df.set_index('close_time')

            # Put the dataframe into postgres sql table
            try:
                df.to_sql(table_name, engine, if_exists="append")
            except Exception as e:
                print(e)


def main():
    """#todo: Document"""
    day_in_milliseconds = 86400 * 1000
    pprint.pprint("Start Script")

    # File for dowloading the data from binance server using rest api

    # logging.basicConfig(level=logging.DEBUG)

    # START_DATE = "10 Aug, 2017"  #7
    # END_DATE = "15 Aug, 2017"  #15
    # FREQUENCY = "5m"  #minutes
    # DATABASE_DETAILS = [DB_HOST, DB_PASSWORD, DB_USER, DB_NAME]
    # SELECTED_SYMBOLS = ["ETHBTC", "LTCBTC", "BNBBTC", "NEOBTC", "FAKESYMBOL"]
    # get_restapi_klines(START_DATE, END_DATE, FREQUENCY, SELECTED_SYMBOLS,
    #                    DATABASE_DETAILS)

    DATABASE_DETAILS = [DB_HOST, DB_PASSWORD, DB_USER, DB_NAME]
    START_DATE = "1 May, 2019"  # 7
    END_DATE = "1 Jun, 2019"  # 15
    # SELECTED_SYMBOLS = ["ETHBTC", "LTCBTC", "BNBBTC", "FAKESYMBOL"]
    SELECTED_SYMBOLS = ["BTCUSDT"]
    get_rest_api_trade_data(START_DATE, END_DATE, SELECTED_SYMBOLS, DATABASE_DETAILS)


# # Reading datasets
# filenames = ['test.tfrecord']
# raw_dataset = tf.data.TFRecordDataset(filenames)

# read_features = {
#     'Aggregate tradeId': tf.io.FixedLenFeature([], tf.int64),
#     'Price': tf.io.FixedLenFeature([], tf.float32),
#     'Quantity': tf.io.FixedLenFeature([], tf.float32),
#     'First tradeId': tf.io.FixedLenFeature([], tf.int64),
#     'Last tradeId': tf.io.FixedLenFeature([], tf.int64),
#     'Timestamp': tf.io.FixedLenFeature([], tf.int64),
#     'Was Buyer Maker': tf.io.FixedLenFeature([], tf.int64),
#     'Was Trade Best Price': tf.io.FixedLenFeature([], tf.int64)
# }

# def _parse_function(example_proto):
#     # Parse the input tf.Example proto using the dictionary above.
#     return tf.io.parse_single_example(example_proto, read_features)

# parsed_dataset = raw_dataset.map(_parse_function)

# for parsed_record in parsed_dataset.take(10):
#     pprint.pprint(repr(parsed_record))

# TO DO:
# get the code to update live with either the rest api or the socket (use jeffries code for a template)
# could possible just run this code once then get the new data with the web socket (use a batch file to run one then the other)

# - If we are going to use the order book we will need to use the socket api because we can't get it into the past
# - Could use the rest api for the trade data/regular candlesticks then add in the market depth from now
# - Get the tf record thing working with the socket api
# -

if __name__ == "__main__":
    main()
