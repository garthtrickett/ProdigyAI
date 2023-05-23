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
