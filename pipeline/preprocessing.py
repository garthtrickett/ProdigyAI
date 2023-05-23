print("script initiated")
import time
very_start = time.time()

import getpass
user = getpass.getuser()

import numba
from numba import njit, prange
import argparse
import os
import sys
import tables as tb
sys.path.append("..")
cwd = os.getcwd()
from pathlib import Path

home = str(Path.home())
sys.path.append(home + "/ProdigyAI")

from multiprocessing import cpu_count

import matplotlib.pyplot as plt

import h5py
import numpy as np
np.set_printoptions(suppress=True)  # don't use scientific notati
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
# from hanging_threads import start_monitoring
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, StandardScaler

import keras

import third_party_libraries.finance_ml
import third_party_libraries.hudson_and_thames.mlfinlab as ml
import third_party_libraries.hudson_and_thames.mlfinlab.labeling.labeling as labeling
import third_party_libraries.hudson_and_thames.mlfinlab.sample_weights.attribution as attribution
from third_party_libraries.hudson_and_thames.mlfinlab.labeling import trend_scanning_labels
import third_party_libraries.snippets as snp
from third_party_libraries.finance_ml.stats.vol import *
from library.core import *

# monitoring_thread = start_monitoring(seconds_frozen=360, test_interval=100)

# import googlecloudprofiler

# try:
#     googlecloudprofiler.start(
#         service="preemp-cpu-big-full-jeff_in-max",
#         # verbose is the logging level. 0-error, 1-warning, 2-info,
#         # 3-debug. It defaults to 0 (error) if not set.
#         verbose=3,
#     )
# except (ValueError, NotImplementedError) as exc:
#     print(exc)  # Handle errors here

arg_parse_stage = None

# Sorting out whether we are using the ipython kernel or not
try:
    get_ipython()
    check_if_ipython = True
    path_adjust = "../"

except Exception as e:
    check_if_ipython = False

    split_cwd = cwd.split("/")
    last_string = split_cwd.pop(-1)
    cwd = cwd.replace(last_string, "")
    os.chdir(cwd)

    parser = argparse.ArgumentParser(description="Preprocessing")
    parser.add_argument("-s",
                        "--stage",
                        type=str,
                        help="Stage of Preprocesssing")
    parser.add_argument("-m",
                        "--model",
                        type=str,
                        help="one_model or two_model")
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
    if args.stage != None:
        arg_parse_stage = 1
        if int(args.stage) == 1:
            if os.path.exists(path_adjust + "temp/data_name_gpu.txt"):
                os.remove(path_adjust + "temp/data_name_gpu.txt")
                print("removed temp/data_name_gpu.txt")
            else:
                print("The file does not exist")

    if args.model != None:
        model = args.model
    path_adjust = ""

if cwd == home + "/":
    cwd = cwd + "/ProdigyAI"
    path_adjust = cwd

try:
    with open(path_adjust + "temp/data_name_gpu.txt", "r") as text_file:
        gpu_file_name = text_file.read()
        stage = 2
except:
    stage = 1

side = None

if arg_parse_stage == 1:
    stage = int(args.stage)
print("the stage" + str(stage))

# Overide model and stage for testing
model = "two_model"
stage = 1
print("the overidden stage" + str(stage))

if stage == 2:
    # size
    h5f = h5py.File("data/gpu_output/" + gpu_file_name + ".h5", "r")
    X = h5f["X"][:]
    P = h5f["P"][:]
    sample_weights = h5f["sample_weights"][:]
    sampled_idx_epoch = h5f["sampled_idx_epoch"][:]
    h5f.close()
    data = pq.read_pandas("data/gpu_output/" + gpu_file_name +
                          "_data.parquet").to_pandas()
    X_for_all_labels = data.dropna(subset=["bins"])
    sampled_idx = pd.DatetimeIndex(sampled_idx_epoch)
    X_for_all_labels["predicted_bins"] = P
    side = X_for_all_labels["predicted_bins"]
    # Could use the probabilities (instead of [1,0,0] use [0.2,0.55,0.25]

import yaml
import wandb

yaml_path = path_adjust + "yaml/preprocessing.yaml"
with open(yaml_path) as file:
    yaml_dict = yaml.load(file, Loader=yaml.FullLoader)

config_dictionary = dict(yaml=yaml_path, params=yaml_dict)

wandb.init(
    dir="/home/" + user + "/ProdigyAI/",
    project="prodigyai",
    config=config_dictionary,
)

minimum_return = eval(wandb.config['params']['minimum_return']['value'])
vertical_barrier_seconds = eval(
    wandb.config['params']['vertical_barrier_seconds']['value'])

volume_max = (
    minimum_return + wandb.config['params']['vol_max_modifier']['value']
)  # The higher this is the more an increase in volatility requries an increase
# in return to be considered buy/sell (Increasing this increases end barrier vertical touches)
volume_min = minimum_return + wandb.config['params']['vol_min_modifier'][
    'value']

filter_type = wandb.config['params']['filter_type']['value']

if filter_type == "cm":
    cusum_filter_vol_modifier = wandb.config['params'][
        'cusum_filter_volume_modifier']['value']
else:
    cusum_filter_vol_modifier = 0

use_sample_weights = wandb.config['params']['use_sample_weights']['value']
use_sample_weights = wandb.config['params']['use_fractional_differentiation'][
    'value']

input_type = wandb.config['params']['input_type']['value']

labelling_method = wandb.config['params']['labelling_method']['value']

pair = wandb.config['params']['pair']['value']

window_length = wandb.config['params']['window_length']['value']
horizon = wandb.config['params']['window_length']['value']

# Type of scaling to apply
scaling_type = wandb.config['params']['scaling_type']['value']
# min max limits
minimum = wandb.config['params']['scaling_maximum']['value']
maximum = wandb.config['params']['scaling_minimum']['value']

# parameters["ntb"] = True  # non time bars

# if parameters["ntb"] == True:
#     # Pick whether you want to add in the time since last bar input feature
#     # time since last bar column
#     parameters["tslbc"] = True  # time since last bar column
# else:
#     # Pick whether you want to add in the volume input feature
#     parameters["vbc"] = True  # volume bar column

# Create the txt file string
parameter_string = wandb.run.id

pt_sl = [
    wandb.config['params']['profit_taking_multiplier']['value'],
    wandb.config['params']['stop_loss_multiplier']['value']
]
cpus = cpu_count() - 1

generate_features_and_labels = wandb.config['params'][
    'generate_features_and_labels']['value']
if generate_features_and_labels == True:
    # READ THE DATA
    if stage == 1:
        # Side
        print("starting data load")
        head = wandb.config['params']['head']['value']

        # # read parquet file of dollar bars
        if input_type == "bars":
            # Mlfinlab bars
            data = pq.read_pandas(
                path_adjust + "data/bars/"
                "btcusdt_agg_trades_50_volume_bars.parquet").to_pandas()
            data = data.drop(columns=[
                "open",
                "high",
                "low",
                # "volume",
                "seconds_since_last_bar",
            ])
            # 1 min ohlcv ready made bars
            # data = pq.read_pandas("data/bars/BTCUSDT_1m.parquet").to_pandas()
            # data["date_time"] = pd.to_datetime(data["date_time"], unit='ms')
            if head > 0:
                data = data.head(head)
            data = data.set_index("date_time")
            data.index = pd.to_datetime(data.index, infer_datetime_format=True)

        # read parquet file of raw ticks
        if input_type == "ticks":
            data = pq.read_pandas(
                path_adjust + "data/bars/" +
                "btcusdt_agg_trades_raw_tick_data.parquet").to_pandas()
            data = data.rename(columns={
                "date": "date_time",
                "price": "close",
                "volume": "volume"
            })
            data = data.drop(columns=["volume"])

            if head > 0:
                data = data.head(head)

            data = data.set_index("date_time")
            print("converting index to date_time")
            data.index = pd.to_datetime(data.index,
                                        format="%m/%d/%Y %H:%M:%S.%f")
            print("index converted")
            # Should do something else than drop the duplicates (maybe trades doesnt have duplicate indexes rather than aggtrades)
            data = data.loc[~data.index.duplicated(keep="first")]

        if input_type == "orderbook":

            with open(path_adjust + "temp/orderbook_data_name.txt",
                      "r") as text_file:
                orderbook_preprocessed_file_name = text_file.read()

            h5f = h5py.File(
                path_adjust + "data/orderbook_preprocessed/" +
                orderbook_preprocessed_file_name + "_" + pair + ".h5",
                "r",
            )
            volumes_index_as_epoch = h5f["volumes_index_as_epoch"][:]
            volumes_np_array = h5f["volumes_np_array"][:]
            df_index_as_epoch = h5f["df_index_as_epoch"][:]
            df_np_array = h5f["df_np_array"][:]
            h5f.close()

            volumes = pd.DataFrame(data=volumes_np_array,
                                   index=volumes_index_as_epoch)
            if len(str(volumes.index[0])) == 19:
                volumes.index = pd.to_datetime(volumes.index, unit="ns")
            else:
                volumes.index = pd.to_datetime(volumes.index, unit="ms")

            data = pd.DataFrame(data=df_np_array, index=df_index_as_epoch)

            if len(str(data.index[0])) == 19:
                data.index = pd.to_datetime(data.index, unit="ns")
            else:
                data.index = pd.to_datetime(data.index, unit="ms")
            data.columns = ["close"]
            data.index.name = "date_time"

            if head > 0:
                data = data.head(head)
                volumes = volumes.head(head)
                volumes_index_as_epoch = volumes_index_as_epoch[:head]
                df_np_array = df_np_array[:5000]
                volumes_index_as_epoch[:5000]
                volumes_np_array[:5000]

        print("data load finished")

        # df_np_array
        # df_index_as_epoch
        # minimum_return = 0.000125

        if labelling_method == "trend_scanning":
            # eem_close = pd.read_csv('stock_prices.csv',
            #                         index_col=0,
            #                         parse_dates=[0])
            # eem_close = eem_close['EEM'].loc[pd.Timestamp(2008, 4, 1):pd.
            #                                  Timestamp(2008, 10, 1)]
            # t_events = eem_close.index

            tr_scan_labels = trend_scanning_labels(data.close,
                                                   data.close.index,
                                                   look_forward_window=20,
                                                   min_sample_length=10)
            tr_scan_labels = tr_scan_labels.replace([np.inf, -np.inf], 0)

            tr_scan_labels["t_value"]
            y = tr_scan_labels["t_value"].values
            split = 3
            y[y < -split] = -split - 1
            mask = ((y >= -split) & (y <= split))
            y[mask] = 1
            y[y > split] = 2
            y[y == -split - 1] = 0
            print(np.unique(y, return_counts=True))

            tr_scan_labels = tr_scan_labels.iloc[window_length:]

            labels_index = tr_scan_labels.index

        if labelling_method == "deeplob":
            labels, first_label_integer_index, last_label_integer_index = get_labels_by_deeplob_method(
                df_np_array, horizon, minimum_return)

            y = labels

            labels_index = df_index_as_epoch[
                first_label_integer_index:last_label_integer_index]

            if len(str(labels_index[0])) == 13:
                unit = "ms"
            else:
                unit = "ns"

            labels_index = pd.to_datetime(labels_index, unit=unit)

            print(np.unique(labels, return_counts=True))

        # Checking for duplicates
        # duplicate_fast_search(data.index.duplicated())

        # Fractional differentiation
        if use_sample_weights == "on":
            data_series = data["close"].to_frame()
            # # generate 100 points
            # nsample = 1000

            # ## simulate a simple sinusoidal function
            # x1 = np.linspace(0, 10, nsample)
            # y = pd.Series(1*np.sin(2.0 * x1 + .5))
            # y.plot()
            # c_constant = 1.
            # y_shifted = (y + c_constant).cumsum().rename('Shifted_series').to_frame()
            # y_shifted.plot()

            # df = y_shifted
            # # df=(df-df.mean())/df.std()
            # df['Shifted_series'][1:] = np.diff(df['Shifted_series'].values)
            # df['Shifted_series'].plot()

            kwargs = None
            # data_series = np.log(data_series)  ## is it good to log this?
            frac_diff_series, d = get_opt_d(  # reduces the number of rows and ends up with less vertical barriers touched
                data_series,
                ds=None,
                maxlag=None,  # If we use raw tick data need at least head > 8000
                thres=1e-5,
                max_size=10000,
                p_thres=1e-2,
                autolag=None,
                verbose=1,
            )

            data["close"] = frac_diff_series
            data = data.dropna(subset=["close"])

        data["window_volatility_level"] = np.nan

        start = time.time()
        volatility_level_array = volatility_levels_numba(
            np.ascontiguousarray(data.close.values),
            wandb.config['params']['window_length']['value'])
        data["window_volatility_level"] = volatility_level_array

        # Should adjust the max value
        # To get more vertical touches we can
        # either increase vol_max or
        # decrease the window seconds
        scaler = MinMaxScaler(feature_range=(volume_min,
                                             volume_max))  # normalization

        normed_window_volatility_level = scaler.fit_transform(
            data[["window_volatility_level"]])
        data["window_volatility_level"] = normed_window_volatility_level  #

        end = time.time()
        print(end - start)

        # CUSUM FILTER
        volatility_threshold = data["window_volatility_level"].mean()

        close_copy = data.dropna().close.copy(deep=True)
        close_np_array, close_index_np_array = pandas_series_to_numba_ready_np_arrays(
            close_copy)

        volatility_threshold = volatility_threshold * cusum_filter_vol_modifier
        print("data_len = " + str(len(data)))
        start = time.time()
        sampled_idx = filter_events(
            data,
            close_np_array,
            close_index_np_array,
            volatility_threshold,
            filter_type,
        )
        print("sampled_idx_len = " + str(len(sampled_idx)))
        end = time.time()
        print(end - start)

    if stage == 2:
        # size
        start = time.time()
        volatility_level_array = volatility_levels_numba(
            data.close.values,
            wandb.config['params']['window_length']['value'])
        data["window_volatility_level"] = volatility_level_array

    if labelling_method == "triple_barrier":
        # This code runs for both first and second stage preprocessing
        start = time.time()
        vertical_barrier_timestamps = ml.labeling.add_vertical_barrier(
            t_events=sampled_idx,
            close=data["close"],
            num_seconds=vertical_barrier_seconds)
        end = time.time()
        print("vertical barrier" + str(end - start))

        start = time.time()

        print("Getting triple barrier events")
        first_touch_dates, events, close, side_prediction, pt_sl = ml.labeling.get_first_touch_dates(
            close=data["close"],
            t_events=sampled_idx,
            pt_sl=pt_sl,
            target=data["window_volatility_level"],
            min_ret=minimum_return,
            num_threads=cpus * 2,
            vertical_barrier_times=vertical_barrier_timestamps,
            side_prediction=side,
            split_by=wandb.config['params']['split_by_get_first_touch_dates']
            ['value']  # maybe we want this as large as we can while still fitting in ram
        )

        triple_barrier_events = ml.labeling.get_events_from_first_touch_dates(
            first_touch_dates, events, close, wandb.config['params']
            ['split_by_get_events_from_first_touch_dates']['value'],
            side_prediction, pt_sl)

        end = time.time()
        print("triple_barrier_events finished taking" + str(end - start))
        very_end = time.time()

        start_time = time.time()
        print("Returning Bins")
        labels = ml.labeling.get_bins(triple_barrier_events, data["close"])
        labels = ml.labeling.drop_labels(labels)
        label_counts = labels.bin.value_counts()
        print("label_counts" + str(label_counts))
        end_time = time.time()

        labels_index = labels.index

        print("returning bins finished taking" + str(end_time - start_time))
        # unique, counts = np.unique(y, return_counts=True)

        sampled_idx_epoch = sampled_idx.astype(np.int64) // 1000000
        h5f = h5py.File(
            path_adjust +
            "data/inputs_and_barrier_labels/sampled_idx_epoch.h5", "w")
        h5f.create_dataset("sampled_idx_epoch", data=sampled_idx_epoch)
        h5f.close()

        # save data dataframe
        table = pa.Table.from_pandas(labels)
        pq.write_table(
            table,
            path_adjust + "data/inputs_and_barrier_labels/labels.parquet",
            use_dictionary=True,
            compression="snappy",
            use_deprecated_int96_timestamps=True,
        )

        # save data dataframe
        table = pa.Table.from_pandas(data)
        pq.write_table(
            table,
            path_adjust + "data/inputs_and_barrier_labels/data.parquet",
            use_dictionary=True,
            compression="snappy",
            use_deprecated_int96_timestamps=True,
        )

        # save data dataframe
        table = pa.Table.from_pandas(triple_barrier_events)
        pq.write_table(
            table,
            path_adjust +
            "data/inputs_and_barrier_labels/triple_barrier_events.parquet",
            use_dictionary=True,
            compression="snappy",
            use_deprecated_int96_timestamps=True,
        )
        y_dataframe = labels["bin"]
        data["bins"] = labels["bin"]
        y = np.asarray(y_dataframe)

else:
    labels = pq.read_pandas(
        path_adjust +
        "data/inputs_and_barrier_labels/labels.parquet").to_pandas()
    data = pq.read_pandas(
        path_adjust +
        "data/inputs_and_barrier_labels/data.parquet").to_pandas()
    triple_barrier_events = pq.read_pandas(
        path_adjust +
        "data/inputs_and_barrier_labels/triple_barrier_events.parquet"
    ).to_pandas()

    with open(path_adjust + "temp/orderbook_data_name.txt", "r") as text_file:
        orderbook_preprocessed_file_name = text_file.read()

    h5f = h5py.File(
        path_adjust + "data/orderbook_preprocessed/" +
        orderbook_preprocessed_file_name + "_" + pair + ".h5", "r")
    volumes_index_as_epoch = h5f["volumes_index_as_epoch"][:]
    volumes_np_array = h5f["volumes_np_array"][:]
    h5f.close()

    volumes = pd.DataFrame(data=volumes_np_array, index=volumes_index_as_epoch)
    volumes.index = pd.to_datetime(volumes.index, unit="ms")

    h5f = h5py.File(
        path_adjust + "data/inputs_and_barrier_labels/sampled_idx_epoch.h5",
        "r")
    sampled_idx_epoch = h5f["sampled_idx_epoch"][:]
    h5f.close()
    sampled_idx = pd.DatetimeIndex(sampled_idx_epoch)
    y_dataframe = labels["bin"]
    data["bins"] = labels["bin"]
    y = np.asarray(y_dataframe)
if wandb.config['params']['apply_train_test_split_and_normalize'][
        'value'] == True:
    if stage == 1:
        # Get why from labels

        start_time = time.time()

        # side
        X_for_all_labels = data.loc[labels_index, :]

        if scaling_type == "min_max":
            scaling_type = 0
        elif scaling_type == "z_score":
            scaling_type = 1

        end_time = time.time()
        print(end_time - start_time)
        if wandb.config['params']['highway_rnn']['value'] == True:
            ### FOR HIGHWAY RNN
            X = np.asarray(volumes.loc[labels_index, :])
            maxes_or_means_np_array_train, mins_or_stds_np_array_train = get_fit_scalars(
                scaling_type, X)
            X_normalized = scale_input_features(
                scaling_type,
                maxes_or_means_np_array_train,
                mins_or_stds_np_array_train,
                X,
                minimum,
                maximum,
            )

            h5f = h5py.File(
                path_adjust + "data/preprocessed/" + parameter_string +
                "_gam_rhn.h5", "w")
            h5f.create_dataset("X", data=X_normalized)
            h5f.create_dataset("y", data=y)
            h5f.close()
        if wandb.config['params']['deeplob_or_tabl']['value'] == True:
            ### One hot encode y
            y = keras.utils.to_categorical(
                y, num_classes=wandb.config['params']['num_classes']['value'])
            np.unique(y[:, 0], return_counts=True)

            start_time = time.time()

            prices_for_window = data.loc[X_for_all_labels.index]
            prices_for_window_index = prices_for_window.index.astype(
                np.int64) // 1000000
            prices_for_window_index_array = np.asarray(prices_for_window_index)

            end_time = time.time()
            print(end_time - start_time)

            start_time = time.time()

            close_index = data.close.index.astype(np.int64) // 1000000
            close_index_array = np.asarray(close_index)

            # Trim the extra bit off the end so prices_for_window and close match up
            close_index_array, end_index = trim_close_index_array(
                close_index_array, prices_for_window_index_array)

            end_time = time.time()
            print(end_time - start_time)

            start_time = time.time()

            # Make a new column time since last bar
            unindexed_data = data.reset_index()
            unindexed_data["shifted_date_time"] = unindexed_data[
                "date_time"].shift(1)
            unindexed_data["time_since_last_bar"] = (
                unindexed_data["date_time"].sub(
                    unindexed_data["shifted_date_time"], axis=0).dt.seconds)
            unindexed_data = unindexed_data.set_index("date_time")
            data["time_since_last_bar"] = unindexed_data["time_since_last_bar"]
            data["time_since_last_bar"].iloc[0] = 0

            end_time = time.time()
            print(end_time - start_time)

            start_time = time.time()

            ### ORDERBOOK VOLUME DATA
            volumes_for_all_labels = volumes.loc[data.close.index][:end_index +
                                                                   1]

            # ## TRADE DATA
            # input_features_trade = []
            # close_array = data.close.values
            # input_features_trade.append(close_array)

            # if parameters["ntb"] == False and parameters["vbc"] == True:
            #     volume_array = data.volume.values
            #     input_features_trade.append(volume_array)
            # if parameters["ntb"] == True and parameters["tslbc"] == True:
            #     time_since_last_bar_array = data.time_since_last_bar.values
            #     input_features_trade.append(time_since_last_bar_array)

            end_time = time.time()
            print(end_time - start_time)

            print("starting splitting")

            prices_for_window_index_array_train_and_val, close_index_array_train_and_val, prices_for_window_index_array_train, close_index_array_train, prices_for_window_index_array_val, close_index_array_val, prices_for_window_index_array_test, close_index_array_test, input_features_end_index_train_and_val, input_features_start_index_train, input_features_end_index_train, input_features_start_index_val, input_features_end_index_val, input_features_start_index_test, y_train, y_val, y_test = split_train_validation_and_test_prices_for_window_and_close_index_array(
                prices_for_window_index_array, close_index_array,
                window_length, y)

            # Mid prices train
            mid_prices_train = data.loc[pd.to_datetime(
                prices_for_window_index_array_train, unit="ms")].close.values

            # Mid prices val
            mid_prices_val = data.loc[pd.to_datetime(
                prices_for_window_index_array_val, unit="ms")].close.values

            # Mid prices test
            mid_prices_test = data.loc[pd.to_datetime(
                prices_for_window_index_array_test, unit="ms")].close.values

            print("starting finished")

            ### Normalizing using last N days before splitting

            if wandb.config['params']['use_last_n_days_scaling'][
                    'value'] == True:

                print("make_input_features_from_orderbook_data started")

                # Make input features from orderbook data
                input_features = make_input_features_from_orderbook_data(
                    volumes_for_all_labels)

                print("make_input_features_from_orderbook_data finishedd")

                psuedo_day_length_in_seconds = wandb.config['params'][
                    'psuedo_day_length_in_seconds'][
                        'value']  # 86400 would be a day

                print("get_integer_indexes_for_last_n_day_scaling started")

                start_and_finish_indexes = get_integer_indexes_for_last_n_day_scaling(
                    input_features, psuedo_day_length_in_seconds,
                    close_index_array)

                print("get_integer_indexes_for_last_n_day_scaling finished")

                print("normalize_based_on_past_n_days_stats started")

                input_features_normalized = normalize_based_on_past_n_days_stats(
                    start_and_finish_indexes, input_features, scaling_type,
                    minimum, maximum)

                print("normalize_based_on_past_n_days_stats finished")

                print("splitting input features started")
                input_features_normalized_train_and_val = input_features_normalized[:, :
                                                                                    input_features_end_index_train_and_val]
                input_features_normalized_train = input_features_normalized_train_and_val[:,
                                                                                          input_features_start_index_train:
                                                                                          input_features_end_index_train]

                input_features_normalized_val = input_features_normalized_train_and_val[:,
                                                                                        input_features_start_index_val:
                                                                                        input_features_end_index_val]

                input_features_normalized_test = input_features_normalized[:,
                                                                           input_features_start_index_test:]

                print("splitting input features finished")

            elif wandb.config['params']['use_last_n_days_scaling'][
                    'value'] == False:

                ### Split intothe training/validation/test sets
                print("splitting into train/va/test sets start")
                start_time = time.time()

                volumes_for_all_labels_train_and_val = volumes_for_all_labels.iloc[:
                                                                                   input_features_end_index_train_and_val]
                volumes_for_all_labels_train = volumes_for_all_labels_train_and_val.iloc[
                    input_features_start_index_train:
                    input_features_end_index_train]
                volumes_for_all_labels_val = volumes_for_all_labels_train_and_val.iloc[
                    input_features_start_index_val:
                    input_features_end_index_val]
                volumes_for_all_labels_test = volumes_for_all_labels.iloc[
                    input_features_start_index_test:]

                end_time = time.time()

                print("splitting into train/va/test sets finished" +
                      str(end_time - start_time))

                print("Make input features from orderbook data started")
                start_time = time.time()

                # Make input features from orderbook data
                input_features_train = make_input_features_from_orderbook_data(
                    volumes_for_all_labels_train)

                end_time = time.time()
                print("Make input features from orderbook data started " +
                      str(end_time - start_time))

                print("Get train scalers started")
                start_time = time.time()
                ### 2: Apply the normalisation based on the training fit scalars (-1,1 vs 0,1 min max test both)
                maxes_or_means_np_array_train, mins_or_stds_np_array_train = get_fit_scalars(
                    scaling_type, input_features_train)

                end_time = time.time()
                print("Get train scalers finished" +
                      str(end_time - start_time))

                print("Norm train data started")
                start_time = time.time()

                # Norm train
                input_features_normalized_train = scale_input_features(
                    scaling_type,
                    maxes_or_means_np_array_train,
                    mins_or_stds_np_array_train,
                    input_features_train,
                    minimum,
                    maximum,
                )

                end_time = time.time()
                print("Get train scalers finished" +
                      str(end_time - start_time))

                # Val

                input_features_val = make_input_features_from_orderbook_data(
                    volumes_for_all_labels_val)

                # Norm val
                input_features_normalized_val = scale_input_features(
                    scaling_type,
                    maxes_or_means_np_array_train,
                    mins_or_stds_np_array_train,
                    input_features_val,
                    minimum,
                    maximum,
                )

                # norm Test
                input_features_test = make_input_features_from_orderbook_data(
                    volumes_for_all_labels_test)

                # Norm test
                input_features_normalized_test = scale_input_features(
                    scaling_type,
                    maxes_or_means_np_array_train,
                    mins_or_stds_np_array_train,
                    input_features_test,
                    minimum,
                    maximum,
                )

    print("plotting started")

    # plot the whole sequence
    # ax = plt.gca()
    # data.plot(y="close", use_index=True)

    # window_index = 500
    # ax = plt.gca()

    # data.iloc[window_index - 200:window_index + 200].plot(y="close",
    #                                                       use_index=True)
    # # plot_window_and_touch_and_label(
    # #     window_index, wandb.config['params']['window_length']['value'], data,
    # #     triple_barrier_events, labels)

    # data.iloc[window_index - 10:window_index + 30]

    data["bins"] = pd.DataFrame(data=labels,
                                index=pd.to_datetime(labels_index, unit="ms"))

    data["downs"] = data.loc[data['bins'] == -1].close
    data["middles"] = data.loc[data['bins'] == 0].close
    data["ups"] = data.loc[data['bins'] == 1].close

    data[["downs", "middles",
          "ups"]].iloc[-10000:].plot(y=["downs", "middles", "ups"],
                                     use_index=True,
                                     style=".")

    print("plotting finished")

    # Sample Weights
    # if stage == 1:
    #     weights = attribution.get_weights_by_return(triple_barrier_events.dropna(),
    #                                                 data.close,
    #                                                 num_threads=5)
    #     sample_weights = np.asarray(weights)
    #     sample_weights = sample_weights.reshape(len(sample_weights))
    #     sampled_idx_epoch = sampled_idx.astype(np.int64) // 1000000 #

    if stage == 2:
        # size
        parameter_string = parameter_string + "second_stage"

    print("writing train/val/test to .h5 files starting")
    start = time.time()

    # np.unique(y_train[:, 0], return_counts=True)[1][1]
    # np.unique(y_train[:, 1], return_counts=True)[1][1]
    # np.unique(y_train[:, 2], return_counts=True)[1][1]

    # np.unique(y_val[:, 0], return_counts=True)[1][1]
    # np.unique(y_val[:, 1], return_counts=True)[1][1]
    # np.unique(y_val[:, 2], return_counts=True)[1][1]

    # np.unique(y_test[:, 0], return_counts=True)[1][1]
    # np.unique(y_test[:, 1], return_counts=True)[1][1]
    # np.unique(y_test[:, 2], return_counts=True)[1][1]

    # Writing preprocessed X,y
    h5f = h5py.File(
        path_adjust + "data/preprocessed/" + parameter_string + ".h5", "w")
    h5f.create_dataset("prices_for_window_index_array_train",
                       data=prices_for_window_index_array_train)
    h5f.create_dataset("prices_for_window_index_array_val",
                       data=prices_for_window_index_array_val)
    h5f.create_dataset("prices_for_window_index_array_test",
                       data=prices_for_window_index_array_test)
    h5f.create_dataset("input_features_normalized_train",
                       data=input_features_normalized_train)
    h5f.create_dataset("input_features_normalized_val",
                       data=input_features_normalized_val)
    h5f.create_dataset("input_features_normalized_test",
                       data=input_features_normalized_test)
    h5f.create_dataset("y_train", data=y_train)
    h5f.create_dataset("y_val", data=y_val)
    h5f.create_dataset("y_test", data=y_test)
    h5f.close()

    end_time = time.time()

    print("writing train/val/test to .h5 files finished taking " +
          str(end_time - start_time))

# if stage == 2:
#     # size
#     h5f.create_dataset("P", data=P)
# h5f.create_dataset("y", data=y)
# h5f.create_dataset("sampled_idx_epoch", data=sampled_idx_epoch)
# if use_sample_weights == "on":
#     h5f.create_dataset("sample_weights", data=sample_weights)
# elif use_sample_weights == "off":
#     h5f.create_dataset("sample_weights", data=np.zeros(1))
# h5f.close()

# # save data dataframe
# table = pa.Table.from_pandas(data)

# pq.write_table(
#     table,
#     path_adjust + "data/preprocessed/" + parameter_string + "_data.parquet",
#     use_dictionary=True,
#     compression="snappy",
#     use_deprecated_int96_timestamps=True,
# )

with open(path_adjust + "temp/data_name.txt", "w+") as text_file:
    text_file.write(parameter_string)
very_end = time.time()
print("full_script_time" + str(very_end - very_start))
print(parameter_string + ".h5")
#