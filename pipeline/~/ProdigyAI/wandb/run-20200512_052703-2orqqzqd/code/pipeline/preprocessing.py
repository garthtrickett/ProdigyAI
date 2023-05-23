print("script initiated")
import time
very_start = time.time()

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

with open(path_adjust + "temp/is_script_finished.txt", "w+") as text_file:
    text_file.write("start_script_time" + str(very_start))

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
    dir="~/ProdigyAI/",
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

regenerate_features_and_labels = True
if regenerate_features_and_labels == True:
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
                orderbook_preprocessed_file_name + ".h5",
                "r",
            )
            volumes_index_as_epoch = h5f["volumes_index_as_epoch"][:]
            volumes_np_array = h5f["volumes_np_array"][:]
            df_index_as_epoch = h5f["df_index_as_epoch"][:]
            df_np_array = h5f["df_np_array"][:]
            h5f.close()

            volumes = pd.DataFrame(data=volumes_np_array,
                                   index=volumes_index_as_epoch)
            volumes.index = pd.to_datetime(volumes.index, unit="ms")
            data = pd.DataFrame(data=df_np_array, index=df_index_as_epoch)
            data.index = pd.to_datetime(data.index, unit="ms")
            data.columns = ["close"]
            data.index.name = "date_time"

            if head > 0:
                data = data.head(head)

        print("data load finished")

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
    triple_barrier_events = ml.labeling.get_events(
        close=data["close"],
        t_events=sampled_idx,
        pt_sl=pt_sl,
        target=data["window_volatility_level"],
        min_ret=minimum_return,
        num_threads=cpus * 2,
        vertical_barrier_times=vertical_barrier_timestamps,
        side_prediction=side,
        split_by=wandb.config['params']['split_by']
        ['value']  # maybe we want this as large as we can while still fitting in ram
    )

    end = time.time()
    print("triple_barrier_events finished taking" + str(end - start))
    very_end = time.time()
    with open(path_adjust + "temp/is_script_finished.txt", "w+") as text_file:
        text_file.write("full_script_time" + str(very_end - very_start))

    start_time = time.time()
    print("Returning Bins")
    labels = ml.labeling.get_bins(triple_barrier_events, data["close"])
    labels = ml.labeling.drop_labels(labels)
    label_counts = labels.bin.value_counts()
    print("label_counts" + str(label_counts))
    end_time = time.time()

    print("returning bins finished taking" + str(end_time - start_time))
    # unique, counts = np.unique(y, return_counts=True)

    sampled_idx_epoch = sampled_idx.astype(np.int64) // 1000000
    h5f = h5py.File(
        path_adjust + "data/inputs_and_barrier_labels/sampled_idx_epoch.h5",
        "w")
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
        orderbook_preprocessed_file_name + ".h5", "r")
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

if stage == 1:
    # Get why from labels
    y_dataframe = labels["bin"]
    data["bins"] = labels["bin"]
    y = np.asarray(y_dataframe)

    start_time = time.time()
    # side
    X_for_all_labels = data.loc[labels.index, :]

    end_time = time.time()
    print(end_time - start_time)

    ### FOR HIGHWAY RNN
    X = np.asarray(volumes.loc[labels.index, :])

    h5f = h5py.File(
        path_adjust + "data/preprocessed/" + parameter_string + "_gam_rhn.h5",
        "w")
    h5f.create_dataset("X", data=X)
    h5f.create_dataset("y", data=y)
    h5f.close()

    X = []

    ### One hot encode y
    y = keras.utils.to_categorical(y, num_classes=3)

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

    end_time = time.time()
    print(end_time - start_time)

    start_time = time.time()

    # Make a new column time since last bar
    unindexed_data = data.reset_index()
    unindexed_data["shifted_date_time"] = unindexed_data["date_time"].shift(1)
    unindexed_data["time_since_last_bar"] = (unindexed_data["date_time"].sub(
        unindexed_data["shifted_date_time"], axis=0).dt.seconds)
    unindexed_data = unindexed_data.set_index("date_time")
    data["time_since_last_bar"] = unindexed_data["time_since_last_bar"]
    data["time_since_last_bar"].iloc[0] = 0

    end_time = time.time()
    print(end_time - start_time)

    start_time = time.time()

    ### ORDERBOOK VOLUME DATA
    volumes_for_all_labels = volumes.loc[data.close.index]

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

    # Type of scaling to apply
    scaling_type = wandb.config['params']['scaling_type']['value']

    # min max limits
    minimum = wandb.config['params']['scaling_maximum']['value']
    maximum = wandb.config['params']['scaling_minimum']['value']

    ### Split intothe training/validation/test sets
    print("splitting into train/va/test sets start")
    start_time = time.time()

    prices_for_window_index_array_train_and_val = prices_for_window_index_array[:round(
        len(prices_for_window_index_array) * 0.8)]

    y_train_and_val = y[:round(len(y) * 0.8)]

    prices_for_window_index_array_train = prices_for_window_index_array_train_and_val[:round(
        len(prices_for_window_index_array_train_and_val) * 0.8)]

    y_train = y_train_and_val[:round(len(y_train_and_val) * 0.8)]

    train_close_array_integer_index = np.nonzero(
        np.in1d(close_index_array, prices_for_window_index_array_train))[0]

    volumes_for_all_labels_train = volumes_for_all_labels.iloc[
        train_close_array_integer_index[0] -
        wandb.config['params']['window_length']['value']:
        train_close_array_integer_index[-1] + 2]

    close_index_array_train = close_index_array[
        train_close_array_integer_index[0] -
        wandb.config['params']['window_length']['value']:
        train_close_array_integer_index[-1] + 2]

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
    print("Get train scalers finished" + str(end_time - start_time))

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
    print("Get train scalers finished" + str(end_time - start_time))

    # print("Make window started")
    # start_time = time.time()

    # padding = wandb.config['params']['window_length']['value'] * 2

    # split_by = 100000

    # number_of_splits = len(prices_for_window_index_array_train) // split_by
    # for i in range(number_of_splits):
    #     print(i)
    #     start_time = time.time()
    #     if i == 0:
    #         start_index = None  # 0
    #         end_index = (i + 1) * split_by
    #         close_and_input_start_index = start_index
    #         close_and_input_end_index = end_index + (padding)
    #     elif i < number_of_splits - 1:
    #         start_index = i * split_by
    #         end_index = (i + 1) * split_by
    #         close_and_input_start_index = start_index - (padding)
    #         close_and_input_end_index = end_index + (padding)
    #     elif i == number_of_splits - 1:
    #         start_index = i * split_by
    #         end_index = None  # -1
    #         close_and_input_start_index = start_index - (padding)
    #         close_and_input_end_index = end_index

    #     # Window train
    #     X_train_section = make_window_multivariate_numba(
    #         len(prices_for_window_index_array_train[start_index:end_index]),
    #         input_features_normalized_train[:, close_and_input_start_index:
    #                                         close_and_input_end_index],
    #         wandb.config['params']['window_length']['value'],
    #         model_arch,
    #     )

    #     print(X_train_section.shape)

    #     hdf5_epath = path_adjust + "data/preprocessed/X_and_y.h5"
    #     if os.path.exists(hdf5_epath) == False or i == 0:
    #         h5f = tb.open_file(hdf5_epath, mode="a")
    #         dataGroup = h5f.create_group(h5f.root, "MyData")
    #         h5f.create_earray(dataGroup,
    #                           "X_train_section",
    #                           obj=X_train_section)
    #         h5f.close()

    #     else:
    #         h5f = tb.open_file(hdf5_epath, mode="r+")
    #         h5f.root.MyData.X_train_section.append(X_train_section)
    #         h5f.close()

    #     # h5f = h5py.File(path_adjust + "data/preprocessed/X_and_y.h5", "w")
    #     # h5f.create_dataset("X_train_section", data=X_train_section)
    #     # h5f.close()

    #     end_time = time.time()

    #     print(str(i) + " finished taking " + str(end_time - start_time))

    # end_time = time.time()
    # print("Make window finished" + str(end_time - start_time))

    #     # import pdb
    #     # pdb.set_trace()

    # Val
    prices_for_window_index_array_val = prices_for_window_index_array_train_and_val[
        round(len(prices_for_window_index_array_train_and_val) * 0.8):]

    y_val = y_train_and_val[round(len(y_train_and_val) * 0.8):]

    val_close_array_integer_index = np.nonzero(
        np.in1d(close_index_array, prices_for_window_index_array_val))[0]

    volumes_for_all_labels_val = volumes_for_all_labels.iloc[
        val_close_array_integer_index[0] -
        wandb.config['params']['window_length']['value']:
        val_close_array_integer_index[-1] + 2]

    close_index_array_val = close_index_array[
        val_close_array_integer_index[0] -
        wandb.config['params']['window_length']['value']:
        val_close_array_integer_index[-1] + 2]

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

    #     # # Make window val
    #     # X_val = make_window_multivariate_numba(
    #     #     prices_for_window_index_array_val,
    #     #     close_index_array_val,
    #     #     input_features_normalized_val,
    #     #     wandb.config['params']['window_length']['value'],
    #     #     model_arch,
    #     # )

    # Test
    prices_for_window_index_array_test = prices_for_window_index_array[
        round(len(prices_for_window_index_array) * 0.8):]

    y_test = y[round(len(y) * 0.8):]

    test_close_array_integer_index = np.nonzero(
        np.in1d(close_index_array, prices_for_window_index_array_test))[0]

    volumes_for_all_labels_test = volumes_for_all_labels.iloc[
        test_close_array_integer_index[0] -
        wandb.config['params']['window_length']['value']:
        test_close_array_integer_index[-1] + 2]

    close_index_array_test = close_index_array[
        test_close_array_integer_index[0] -
        wandb.config['params']['window_length']['value']:
        test_close_array_integer_index[-1] + 2]

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

#     # # Window test
#     # # TABL
#     # X_test = make_window_multivariate_numba(
#     #     prices_for_window_index_array_test,
#     #     close_index_array_test,
#     #     input_features_normalized_test,
#     #     wandb.config['params']['window_length']['value'],
#     #     model_arch,
#     # )

#     start = time.time()

#     end = time.time()
#     print("numba make window time" + str(end - start))
#     start = time.time()

print("plotting started")

# plot the whole sequence
ax = plt.gca()
data.plot(y="close", use_index=True)

window_index = 500
ax = plt.gca()

data.iloc[window_index - 200:window_index + 200].plot(y="close",
                                                      use_index=True)
plot_window_and_touch_and_label(
    window_index, wandb.config['params']['window_length']['value'], data,
    triple_barrier_events, labels)

data.iloc[window_index - 10:window_index + 30]

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

# Writing preprocessed X,y
h5f = h5py.File(path_adjust + "data/preprocessed/" + parameter_string + ".h5",
                "w")
h5f.create_dataset("prices_for_window_index_array_train",
                   data=prices_for_window_index_array_train)
h5f.create_dataset("prices_for_window_index_array_val",
                   data=prices_for_window_index_array_val)
h5f.create_dataset("prices_for_window_index_array_test",
                   data=prices_for_window_index_array_test)
h5f.create_dataset("close_index_array_train", data=close_index_array_train)
h5f.create_dataset("close_index_array_val", data=close_index_array_val)
h5f.create_dataset("close_index_array_test", data=close_index_array_test)
h5f.create_dataset("input_features_normalized_train",
                   data=input_features_normalized_train)
h5f.create_dataset("input_features_normalized_val",
                   data=input_features_normalized_val)
h5f.create_dataset("input_features_normalized_test",
                   data=input_features_normalized_test)
h5f.create_dataset("y_train", data=y_train)
h5f.create_dataset("y_val", data=y_val)
h5f.create_dataset("y_test", data=y_test)

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
