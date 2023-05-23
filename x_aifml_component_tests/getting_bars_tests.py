import os
import sys
sys.path.append("..")
cwd = os.getcwd()
import numpy as np
import pandas as pd

from third_party_libraries.mlfinlab.data_structures import standard_data_structures as ds

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
    path_adjust = ""

threshold = 50

import pandas as pd
df = pd.read_parquet(path_adjust + "data/bars/raw_tick_data.parquet")
df = df.head(1000)
df = df.rename(columns={
    "date": "date_time",
    "price": "price",
    "volume": "volume"
})
df = df.set_index("date_time")
df.to_csv(path_adjust + "data/bars/raw_tick_data.csv")

bars = ds.get_volume_bars(path_adjust + "data/bars/raw_tick_data.csv",
                          threshold=threshold,
                          batch_size=1000,
                          verbose=False)

db1 = ds.get_dollar_bars(path_adjust + "data/bars/raw_tick_data.csv",
                         threshold=threshold,
                         batch_size=1000,
                         verbose=False)

db1 = ds.get_tick_bars(path_adjust + "data/bars/raw_tick_data.csv",
                       threshold=threshold,
                       batch_size=1000,
                       verbose=False)

exp_num_ticks_init = 1000
num_prev_bars = 3
from third_party_libraries.mlfinlab.data_structures import run_data_structures as ds
db1, thresh_1 = ds.get_ema_dollar_run_bars(
    path_adjust + "data/bars/raw_tick_data.csv",
    exp_num_ticks_init=exp_num_ticks_init,
    expected_imbalance_window=10000,
    num_prev_bars=num_prev_bars,
    batch_size=2e7,
    verbose=False,
    analyse_thresholds=True)
db2, thresh_2 = ds.get_ema_dollar_run_bars(
    path_adjust + "data/bars/raw_tick_data.csv",
    exp_num_ticks_init=exp_num_ticks_init,
    expected_imbalance_window=10000,
    num_prev_bars=num_prev_bars,
    batch_size=50,
    verbose=False,
    analyse_thresholds=True)
db3, _ = ds.get_ema_dollar_run_bars(path_adjust +
                                    "data/bars/raw_tick_data.csv",
                                    exp_num_ticks_init=exp_num_ticks_init,
                                    expected_imbalance_window=10000,
                                    num_prev_bars=num_prev_bars,
                                    batch_size=10,
                                    verbose=False)

from third_party_libraries.mlfinlab.data_structures import imbalance_data_structures as ds

db1, _ = ds.get_ema_dollar_imbalance_bars(
    path_adjust + "data/bars/raw_tick_data.csv",
    exp_num_ticks_init=exp_num_ticks_init,
    expected_imbalance_window=10000,
    num_prev_bars=num_prev_bars,
    batch_size=2e7,
    verbose=False,
)
db2, _ = ds.get_ema_dollar_imbalance_bars(
    path_adjust + "data/bars/raw_tick_data.csv",
    exp_num_ticks_init=exp_num_ticks_init,
    expected_imbalance_window=10000,
    num_prev_bars=num_prev_bars,
    batch_size=50,
    verbose=False)
db3, _ = ds.get_ema_dollar_imbalance_bars(
    path_adjust + "data/bars/raw_tick_data.csv",
    exp_num_ticks_init=exp_num_ticks_init,
    expected_imbalance_window=10000,
    num_prev_bars=num_prev_bars,
    batch_size=10,
    verbose=False)

from third_party_libraries.mlfinlab.data_structures import time_data_structures as ds

db1 = ds.get_time_bars(path_adjust + "data/bars/raw_tick_data.csv",
                       resolution='MIN',
                       num_units=1,
                       batch_size=1000,
                       verbose=False)
db2 = ds.get_time_bars(path_adjust + "data/bars/raw_tick_data.csv",
                       resolution='MIN',
                       num_units=1,
                       batch_size=50,
                       verbose=False)
db3 = ds.get_time_bars(path_adjust + "data/bars/raw_tick_data.csv",
                       resolution='MIN',
                       num_units=1,
                       batch_size=10,
                       verbose=False)

bars['date_time'] = bars['date_time'].astype('datetime64[ns]')

bars['seconds_since_last_bar'] = np.nan

row_id = 0
for row in bars.itertuples():
    if row_id > 0:
        bars['seconds_since_last_bar'].iloc[row_id] = (
            bars.iloc[row_id].date_time -
            bars.iloc[row_id - 1].date_time).seconds
    else:
        bars['seconds_since_last_bar'].iloc[row_id] = 0

    row_id = row_id + 1

output_path = "data/bars/btcusdt_agg_trades_50_bars"
bars.to_parquet(output_path)

data = pd.read_csv('data/BTCUSDT_1m.csv')
data = data[["Open time", "Close", "Volume"]]
data = data.rename(columns={
    "Open time": "date_time",
    "Close": "close",
    "Volume": "volume"
})
output_path = "data/bars/BTCUSDT_1m.parquet"
data.to_parquet(output_path)
