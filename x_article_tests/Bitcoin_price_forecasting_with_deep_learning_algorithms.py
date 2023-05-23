import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error
from math import sqrt
from random import randint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers
from matplotlib import pyplot
from datetime import datetime
from matplotlib import pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
from tensorflow.keras.callbacks import TensorBoard

import pyarrow as pa
import pyarrow.parquet as pq

data = pq.read_pandas(
    'data/btcusdt_agg_trades_10_tick_bars.parquet').to_pandas()

df = data.filter(['date_time', 'close'], axis=1)

df = df.set_index('date_time')

df = df.head(5000)


def train_test_split(df, test_size=0.1):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data


def line_plot(line1, line2, label1=None, label2=None, title=''):
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(line1, label=label1, linewidth=2)
    ax.plot(line2, label=label2, linewidth=2)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=18)


train, test = train_test_split(df, test_size=0.1)

working_data = [train, test]
working_data = pd.concat(working_data)

# Seasonal Decomposition
s = sm.tsa.seasonal_decompose(working_data.close.values, freq=60)

trace1 = go.Scatter(x=np.arange(0, len(s.trend), 1),
                    y=s.trend,
                    mode='lines',
                    name='Trend',
                    line=dict(color=('rgb(244, 146, 65)'), width=4))
trace2 = go.Scatter(x=np.arange(0, len(s.seasonal), 1),
                    y=s.seasonal,
                    mode='lines',
                    name='Seasonal',
                    line=dict(color=('rgb(66, 244, 155)'), width=2))

trace3 = go.Scatter(x=np.arange(0, len(s.resid), 1),
                    y=s.resid,
                    mode='lines',
                    name='Residual',
                    line=dict(color=('rgb(209, 244, 66)'), width=2))

trace4 = go.Scatter(x=np.arange(0, len(s.observed), 1),
                    y=s.observed,
                    mode='lines',
                    name='Observed',
                    line=dict(color=('rgb(66, 134, 244)'), width=2))

data = [trace1, trace2, trace3, trace4]
layout = dict(title='Seasonal decomposition',
              xaxis=dict(title='Time'),
              yaxis=dict(title='Price, USD'))
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='seasonal_decomposition')

# Autocorrelation
plt.figure(figsize=(15, 7))
ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(working_data.close.values.squeeze(), lags=48, ax=ax)
ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(working_data.close.values.squeeze(), lags=48, ax=ax)
plt.tight_layout()
plt.show()


# Window making function
def create_lookback(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


from sklearn.preprocessing import MinMaxScaler

training_set = train.values
training_set = np.reshape(training_set, (len(training_set), 1))
test_set = test.values
test_set = np.reshape(test_set, (len(test_set), 1))

#scale datasets
scaler = MinMaxScaler()
training_set = scaler.fit_transform(training_set)
test_set = scaler.transform(test_set)

# create datasets which are suitable for time series forecasting
look_back = 1
X_train, Y_train = create_lookback(training_set, look_back)
X_test, Y_test = create_lookback(test_set, look_back)

# reshape datasets so that they will be ok for the requirements of the LSTM model in Keras
X_train = np.reshape(X_train, (len(X_train), 1, X_train.shape[1]))
X_test = np.reshape(X_test, (len(X_test), 1, X_test.shape[1]))

# Training the model
# initialize sequential model, add 2 stacked LSTM layers and densely connected output neuron
model = Sequential()
model.add(
    LSTM(256,
         return_sequences=True,
         input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(256))
model.add(Dense(1))

# NAME = ("{}-SEQ-{}-PRED-{}".format(SEQ_LEN, FUTURE_PERIOD_PREDICT, TIME))
NAME = "test"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

# compile and fit the model
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train,
                    Y_train,
                    epochs=2,
                    batch_size=16,
                    shuffle=False,
                    validation_data=(X_test, Y_test),
                    callbacks=[
                        EarlyStopping(monitor='val_loss',
                                      min_delta=5e-5,
                                      patience=2,
                                      verbose=1), tensorboard
                    ])

model.save("models/{}".format(NAME))