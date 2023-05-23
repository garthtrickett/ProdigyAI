import sys
import os
import argparse

import numba
from numba import njit, prange

## Path adjustment
sys.path.append("..")
cwd = os.getcwd()
from pathlib import Path
home = str(Path.home())
sys.path.append(home + "/ProdigyAI")

import h5py

import torch

import numpy as np
import pandas as pd
import yaml

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
import wandb

### Ipython and Argparse setup STARTED
try:
    resuming = "NA"
    get_ipython()
    check_if_ipython = True
    path_adjust = "../"
    import getpass

    user = getpass.getuser()
except Exception as e:  ## If not using Ipython kernel deal with any argparse's
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
    parser.add_argument(
        "-f",
        "--resuming",
        type=str,
        help="Is this a continuation of preempted instance?",
    )
    parser.add_argument("-u",
                        "--user",
                        type=str,
                        help="Stage of Preprocesssing")
    args = parser.parse_args()
    if args.user != None:
        user = args.user
    else:
        import getpass
        user = getpass.getuser()
    if args.resuming != None:
        resuming = args.resuming
    else:
        resuming = "NA"
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

### Load yaml config
yaml_path = path_adjust + "yaml/backtesting.yaml"
with open(yaml_path) as file:
    yaml_dict = yaml.load(file, Loader=yaml.FullLoader)

config_dictionary = dict(yaml=yaml_path, params=yaml_dict)

### Load the data
file_name = config_dictionary["params"]["dataset"]["value"]
path = home + "/ProdigyAI/data/preprocessed/" + file_name
h5f = h5py.File(path, "r")
prices_for_window_index_array_test = h5f[
    "prices_for_window_index_array_test"][:]
input_features_normalized_test = h5f["input_features_normalized_test"][:]
y_test = h5f["y_test"][:].astype(np.int8)
mid_prices_test = h5f["mid_prices_test"][:]
h5f.close()

y_test_integers = np.argmax(y_test, axis=1)
data = pd.DataFrame(data=y_test_integers,
                    index=pd.to_datetime(prices_for_window_index_array_test,
                                         unit="ms"))

data["mid_prices"] = mid_prices_test
data.columns = ["label", "mid_prices"]

data["middles"] = data.loc[data['label'] == 0].mid_prices
data["ups"] = data.loc[data['label'] == 1].mid_prices
data["downs"] = data.loc[data['label'] == 2].mid_prices

data[["downs", "middles",
      "ups"]].iloc[0:1000].plot(y=["downs", "middles", "ups"],
                                use_index=True,
                                style=".")

### build a thing that color codes the prices and the labels so I can get an idea of what might work

wandb_id = "wztpxliw"
opt = "adam"
### Loading the model
K.clear_session()
from tensorflow.keras.models import model_from_json
json_path = wandb.restore('model.json',
                          run_path=user + '/prodigyai/' + wandb_id).name
json_file = open(json_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
deeplob = model_from_json(loaded_model_json)
weights_path = wandb.restore('weights.h5',
                             run_path=user + '/prodigyai/' + wandb_id).name
deeplob.load_weights(weights_path)
if opt == "adam":
    opt = Adam(lr=1e-02, beta_1=0.9, beta_2=0.999, epsilon=1)
elif opt == "sgd":
    opt = SGD()
deeplob.compile(optimizer=opt,
                loss="categorical_crossentropy",
                metrics=["accuracy"])

# len(input_features_normalized_test[0])
# len(y_test)


@njit(parallel=True)
def make_x_numba(i, num_features, window_length,
                 input_features_normalized_test):
    X = np.zeros((num_features, window_length))
    X[:] = np.nan
    for j in range(num_features):
        for k in range(window_length):
            X[j][k] = input_features_normalized_test[j][i + k]
    return X


def backtest(y_test, mid_prices_test, input_features_normalized_test):
    original_price = mid_prices_test[0]
    cash = 1
    coin_held = 0
    window_length = input_features_normalized_test.shape[1] - len(y_test)
    num_features = len(input_features_normalized_test)
    for i in range(len(y_test)):  # num samples
        X = make_x_numba(i, num_features, window_length,
                         input_features_normalized_test)
        # Add the batch and end dimension
        X = np.swapaxes(X, 0, 1)
        X = X.reshape((1, X.shape[0], X.shape[1], 1))

        prediction_proba = deeplob.predict(X)
        prediction = np.argmax(prediction_proba, axis=1)
        y_test[i]
        import pdb
        pdb.set_trace()
        if coin_held == 0:
            if prediction == 2:  # up
                coin_held = cash * mid_prices_test[i]
                bought_price = mid_prices_test[i]
                cash = 0
            if prediction == 0:  # down
                coin_held = cash * -mid_prices_test[i]
                bought_price = mid_prices_test[i]
                cash = 0
        if coin_held > 0 and prediction == 0:  ## sell prediction when holding long
            profit_percentage = mid_prices_test[i] / bought_price
            cash = 1 * profit_percentage
            coin_held = 0
        if coin_held < 0 and prediction == 2:  # buy prediction when holding short
            profit_percentage = bought_price / mid_prices_test[i]
            cash = 1 * profit_percentage
            coin_held = 0
        print(cash)
        print(coin_held)


backtest(y_test, mid_prices_test, input_features_normalized_test)

###
### at some point re order the deeplob input and run it again