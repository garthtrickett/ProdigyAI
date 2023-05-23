import pandas as pd
import numpy as np
import numbers
import pyarrow as pa
import pyarrow.parquet as pq

import mlfinlab as ml
from mlfinlab.filters import filters
from mlfinlab.labeling import labeling
from mlfinlab.util import utils
from mlfinlab.features import fracdiff
import library.snippets as snp

data = pq.read_pandas("data/btcusdt_agg_trades_10_tick_bars.parquet").to_pandas()
head = 100000
if head > 0:
    data = data.head(head)

data = data.set_index("date_time")
data.index = pd.to_datetime(data.index)
df = data
# df = df.drop(['open', 'high', 'low', 'volume'], axis=1)
close = df["close"]

vol_span = 100
vol = ml.util.get_daily_vol(close=data["close"], lookback=vol_span)


def cusum_filter(close, h):
    # asssum that E y_t = y_{t-1}
    t_events = []
    s_pos, s_neg = 0, 0
    ret = close.pct_change().dropna()
    diff = ret.diff().dropna()
    # time variant threshold
    if isinstance(h, numbers.Number):
        h = pd.Series(h, index=diff.index)
    h = h.reindex(diff.index, method="bfill")
    h = h.dropna()
    for t in h.index:
        s_pos = max(0, s_pos + diff.loc[t])
        s_neg = min(0, s_neg + diff.loc[t])
        if s_pos > h.loc[t]:
            s_pos = 0
            t_events.append(t)
        elif s_neg < -h.loc[t]:
            s_neg = 0
            t_events.append(t)
    return pd.DatetimeIndex(t_events)


def get_t1(close, t_events, num_days):
    t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=t_events[: t1.shape[0]])
    return t1


def get_3barriers(
    close, t_events, ptsl, trgt, min_ret=0, num_threads=1, t1=False, side=None
):
    # Get sampled target values
    trgt = trgt.loc[t_events]
    trgt = trgt[trgt > min_ret]
    # Get time boundary t1
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=t_events)
    # Define the side
    if side is None:
        _side = pd.Series(1.0, index=trgt.index)
        _ptsl = [ptsl, ptsl]
    else:
        _side = side.loc[trgt.index]
        _ptsl = ptsl[:2]
    events = pd.concat({"t1": t1, "trgt": trgt, "side": _side}, axis=1)
    events = events.dropna(subset=["trgt"])
    time_idx = apply_ptslt1(close, events, _ptsl, events.index)
    # Skip when all of barrier are not touched
    time_idx = time_idx.dropna(how="all")
    events["t1_type"] = time_idx.idxmin(axis=1)
    events["t1"] = time_idx.min(axis=1)
    if side is None:
        events = events.drop("side", axis=1)
    return events


def apply_ptslt1(close, events, ptsl, molecule):
    """Return datafram about if price touches the boundary"""
    # Sample a subset with specific indices
    _events = events.loc[molecule]
    # Time limit

    out = pd.DataFrame(index=_events.index)
    # Set Profit Taking and Stop Loss
    if ptsl[0] > 0:
        pt = ptsl[0] * _events["trgt"]
    else:
        # Switch off profit taking
        pt = pd.Series(index=_events.index)
    if ptsl[1] > 0:
        sl = -ptsl[1] * _events["trgt"]
    else:
        # Switch off stop loss
        sl = pd.Series(index=_events.index)
    # Replace undifined value with the last time index
    time_limits = _events["t1"].fillna(close.index[-1])
    for loc, t1 in time_limits.iteritems():
        df = close[loc:t1]
        # Change the direction depending on the side
        df = (df / close[loc] - 1) * _events.at[loc, "side"]
        # print(df)
        # print(loc, t1, df[df < sl[loc]].index.min(), df[df > pt[loc]].index.min())
        out.at[loc, "sl"] = df[df < sl[loc]].index.min()
        out.at[loc, "pt"] = df[df > pt[loc]].index.min()
    out["t1"] = _events["t1"].copy(deep=True)
    return out


# checks whether would actually get profit
def get_bins(events, close):
    # Prices algined with events
    events = events.dropna(subset=["t1"])
    px = events.index.union(events["t1"].values).drop_duplicates()
    px = close.reindex(px, method="bfill")
    # Create out object
    out = pd.DataFrame(index=events.index)
    out["ret"] = px.loc[events["t1"].values].values / px.loc[events.index] - 1.0
    if "side" in events:
        out["ret"] *= events["side"]
    out["bin"] = np.sign(out["ret"])
    # 0 when touching vertical line
    out["bin"].loc[events["t1_type"] == "t1"] = 0
    if "side" in events:
        out.loc[out["ret"] <= 0, "bin"] = 0
    return out


# Get daily volatility and use it as a threshold for the cusum filter.
sampled_idx = cusum_filter(close, vol)

# Vertical Barrier
t1 = get_t1(close, sampled_idx, num_days=1)

# Triple Barrier + metalabelling
trgt = vol
# events were price has moved enough for vol target
# and has hit either the vertical or horizontal barriers
events = get_3barriers(close, t_events=sampled_idx, trgt=trgt, ptsl=1, t1=t1)

# When the price hit the barrier would that have
# been enough to take the profit we were looking for?
# Ie should we have actually betted.
bins = get_bins(events, close)

bins["bin"].value_counts()

# side can be -1,0,1
# should bet 0,1
