import datetime as dt
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf
from scipy.stats import moment, norm
from sklearn.externals import joblib

from third_party_libraries.mlfinlab.bet_sizing.bet_sizing import (
    bet_size_budget,
    bet_size_dynamic,
    bet_size_probability,
    bet_size_reserve,
    cdf_mixture,
    confirm_and_cast_to_df,
    get_concurrent_sides,
    single_bet_size_mixed,
)
from third_party_libraries.mlfinlab.bet_sizing.ch10_snippets import (
    avg_active_signals,
    bet_size,
    discrete_signal,
    get_signal,
    get_target_pos,
    get_w,
    limit_price,
)
from third_party_libraries.mlfinlab.bet_sizing.ef3m import M2N, most_likely_parameters, raw_moment

scaler = joblib.load("data/gpu_output/scaler_one.save")
stage_one_model = tf.keras.models.load_model("data/gpu_output/my_model_stage_one.h5")
stage_two_model = tf.keras.models.load_model("data/gpu_output/my_model_stage_two.h5")

h5f = h5py.File("data/preprocessed/stage_one.h5", "r")
sample_weights = h5f["sample_weights"][:]
X = h5f["X"][:]
y = h5f["y"][:]
sampled_idx_epoch = h5f["sampled_idx_epoch"][:]
h5f.close()

# h5f = h5py.File("data/preprocessed/stage_two.h5", "r")
# sample_weights = h5f["sample_weights"][:]
# X = h5f["X"][:]
# y = h5f["y"][:]
# sampled_idx_epoch = h5f["sampled_idx_epoch"][:]
# h5f.close()


X = scaler.transform(X)
X = X.reshape(X.shape[0], X.shape[1], 1)
sides = stage_one_model.predict(X)
sides = sides.argmax(axis=-1)
unique, counts = np.unique(sides, return_counts=True)

shifted_sides = np.zeros(len(sides))
shifted_sides[:] = np.nan
row_num = 0
for side in sides:
    if side == 0:
        shifted_sides[row_num] = -1
    elif side == 1:
        shifted_sides[row_num] = 0
    elif side == 2:
        shifted_sides[row_num] = 1

    row_num = row_num + 1

unique, counts = np.unique(shifted_sides, return_counts=True)

sizes = stage_two_model.predict([X, sides.astype(float)])
sizes = [i[0] for i in sizes]
prices = [i[-1] for i in X]
data = pq.read_pandas("data/preprocessed/stage_one.parquet").to_pandas()
t1_data = data.dropna(subset=["t1"])
t1_data = t1_data.drop(columns=["window_volatility_level", "bins"])

t1_data["side"] = shifted_sides
t1_data["prob"] = sizes

t1_data["side"].iloc[0] = -1
t1_data["side"].iloc[1] = 1
t1_data["close"].iloc[0] = 1000
t1_data["close"].iloc[1] = 998


# p0 = 1000, p1 = 998 shorting has profit sell(1000 *(1-0.001)) + buy(-998* (1+0.001))
# p0 = 1000, p1 = 1002 long zero profit buy(-1000 *(1+0.001)) + sell(1003* (1-0.001))

num_classes = 3

# factor in BNB coin price fluctuations
# factor in slippage somehow

cash = 10000
binance_fee = 0.00075  # without_bnb = 0.001
holdings = 0
held_side = None
for i in range(len(t1_data)):
    if t1_data["prob"].iloc[i] > 0.5:
        bet_size = (t1_data["prob"].iloc[i] - 1 / num_classes) / (
            t1_data["prob"].iloc[i] * (1 - t1_data["prob"].iloc[i])
        ) ** 0.5
        bet_size = t1_data["side"][i] * bet_size
        if held_side is None:
            if t1_data["side"][i] != 0:
                print(
                    "open"
                    + str(t1_data["side"].iloc[i])
                    + "position on"
                    + str(i)
                    + "with price"
                    + str(t1_data["close"].iloc[i])
                    + "with bet_size"
                    + str(bet_size)
                )
                held_side = t1_data["side"][i]
                open_position_price = t1_data["close"].iloc[i]
                trade_amount = (cash * bet_size) / open_position_price
                cash_change = trade_amount * open_position_price
                holdings = holdings + trade_amount
                opening_cash = cash
                print(cash)
                if held_side == 1:
                    cash = cash - (cash_change * (1 + binance_fee))
                if held_side == -1:
                    cash = cash - (cash_change * (1 - binance_fee))

        if (
            held_side is not None
            and t1_data["side"][i] != 0
            and held_side != t1_data["side"][i]
        ):
            print(
                "close"
                + str(t1_data["side"].iloc[i])
                + "position on"
                + str(i)
                + "with price"
                + str(t1_data["close"].iloc[i])
            )
            close_position_price = t1_data["close"].iloc[i]
            cash_change = trade_amount * close_position_price
            if held_side == 1:
                cash = cash + (cash_change * (1 - binance_fee))
            if held_side == -1:
                cash = cash + (cash_change * (1 + binance_fee))
            holdings = holdings - trade_amount
            if cash > opening_cash:
                print("Profit Made")
            if cash < opening_cash:
                print("Loss Made")
            print(cash)
            held_side = None
            close_position_price = None
            open_position_price = None
            opening_cash = None

        # successfull short sell: borrow  a stock i don't own and sell
        # it then buy it back at a lower price keeping the difference

bet_size_probabilities = bet_size_probability(
    events=t1_data,
    prob=t1_data["prob"],
    num_classes=3,
    pred=t1_data["side"],
    average_active=True,
    # step_size=0.1,
)

### get_signal == bet_size_probability
### get_signal + avg_active_signals == bet_size_probability' with 'average_active == true'
### get_signal + discrete_signal == bet_size_probability' with 'step_size' greater than 0.
### get_target_pos + limit_price + bet_size  == bet_size_dynamic


# Bet size Dynamic
# Setup the test DataFrame.
dates_test = np.array(
    [dt.datetime(2000, 1, 1) + i * dt.timedelta(days=1) for i in range(5)]
)
events_test = pd.DataFrame(
    data=[
        [25, 55, 75.50, 80.00],
        [35, 55, 76.90, 75.00],
        [45, 55, 74.10, 72.50],
        [40, 55, 67.75, 65.00],
        [30, 55, 62.00, 70.80],
    ],
    columns=["pos", "max_pos", "m_p", "f"],
    index=dates_test,
)
# Calculate results.
d_events = {col: events_test[col] for col in list(events_test.columns)}
events_results = confirm_and_cast_to_df(d_events)
w_param = get_w(10, 0.95, "sigmoid")
events_results["t_pos"] = events_results.apply(
    lambda row: get_target_pos(w_param, row.f, row.m_p, row.max_pos, "sigmoid"), axis=1
)
events_results["l_p"] = events_results.apply(
    lambda row: limit_price(row.t_pos, row.pos, row.f, w_param, row.max_pos, "sigmoid"),
    axis=1,
)
events_results["bet_size"] = events_results.apply(
    lambda row: bet_size(w_param, row.f - row.m_p, "sigmoid"), axis=1
)
df_result = events_results[["bet_size", "t_pos", "l_p"]]
# Evaluate.
result = bet_size_dynamic(
    events_test["pos"], events_test["max_pos"], events_test["m_p"], events_test["f"]
)
