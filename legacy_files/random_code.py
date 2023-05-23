# blackarbs metalabelling
events = get_events(
    close=data["close"],
    timestamps=sampled_idx,
    trgt=volatility_level,
    sltp=pt_sl,
    t1=vertical_barrier_timestamps,
    side=None,
)
triple_barrier_events = events
labels = CoreFunctions.get_bins(triple_barrier_events, data["close"])
labels.bin.value_counts()
zeros = labels[labels.bin == -1]
np.arr
# basic loop currenctly
def add_volatility_level_column(WINDOW_LONG, data):
    for i in range(parameters["WINDOW_LONG"], len(data), 1):
        window = data.iloc[i - parameters["WINDOW_LONG"] : i + 1]
        window_abs_returns = np.abs(window.close.pct_change())
        window_volatility_level = np.std(window_abs_returns) + np.mean(
            window_abs_returns
        )
        data["window_volatility_level"].iloc[i] = window_volatility_level
    return data


def volatility_iterator(index, WINDOW_LONG, data):  #
    row_number = data.index.get_loc(index)
    if row_number >= WINDOW_LONG:
        window = data.iloc[row_number - WINDOW_LONG : row_number + 1]
        window_abs_returns = np.abs(window.close.pct_change())
        window_volatility_level = np.std(window_abs_returns) + np.mean(
            window_abs_returns
        )
        return window_volatility_level


start = time.time()
for index, row in data.iterrows():
    window_volatility_level = volatility_iterator(
        index, parameters["WINDOW_LONG"], data
    )
end = time.time()
print(end - start)


start = time.time()
for row in data.itertuples(index=True):
    window_volatility_level = volatility_iterator(
        row[0], parameters["WINDOW_LONG"], data
    )
end = time.time()
print(end - start)

start = time.time()
volatility_level = data.apply(
    lambda row: volatility_iterator(row.name, parameters["WINDOW_LONG"], data), axis=1
)
end = time.time()
print(end - start)
blah.iloc[-1000:-995]


def get_vertical_barrier_timestamps(data, sampled_idx, seconds=3600):
    delta = pd.Timedelta(seconds=seconds)
    vertical_barrier_list = []
    for row in data.loc[sampled_idx].itertuples(index=True):
        if ((row[0] + delta)) < data.index[-1]:
            nearest = data[(row[0] + delta) :].index[0]
            vertical_barrier = [row[0], nearest]
            vertical_barrier_list.append(vertical_barrier)
    vertical_barrier_dataframe = pd.DataFrame.from_records(
        vertical_barrier_list, columns=["date_time", "vertical_barrier"]
    )
    vertical_barrier_dataframe = vertical_barrier_dataframe.set_index("date_time")
    return vertical_barrier_dataframe["vertical_barrier"]


start = time.time()
print("Computing vertical barriers")
vertical_barrier_timestamps = get_vertical_barrier_timestamps(
    data, sampled_idx, seconds=parameters["vertical_barrier_seconds"]
)
end = time.time()
print(end - start)


window_np_array = data.loc[X_for_all_labels.index].normed_close.values


@njit
def make_window(window_np_array, WINDOW_LONG):
    res = np.zeros((len(window_np_array), WINDOW_LONG, 1))
    res[:] = np.nan
    for i in range(len(window_np_array)):
        window = window_np_array[i - 100 : i]
        window.reshape(WINDOW_LONG, 1)
        res[i] = window
    return res


X = make_window(window_np_array, parameters["WINDOW_LONG"])


# Tuning the volatility levels
print("Tuning Volatility Levels")
start = time.time()
volatility_level = data.window_volatility_level
volatility_level *= 10

up_first = 0
while volatility_level.min() < parameters["min_ret"]:
    up_first = 1
    volatility_level *= 1.05

while volatility_level.min() > parameters["min_ret"] and up_first == 0:
    volatility_level *= 0.95

while volatility_level.min() < parameters["min_ret"]:
    volatility_level *= 1.01
end = time.time()
print(end - start)



# Understanding the triple stuff
# If the volatility is higher than 
# the minimum return keep that sample
target = target.loc[t_events]
target = target[target > min_ret]  
# otherwise drop it
events = pd.concat(
        {"t1": vertical_barrier_times, "trgt": target, "side": side_}, axis=1
    )
    events = events.dropna(subset=["trgt"])

# Set the profit requirement equal to the volatility
profit_taking = profit_taking_multiple * events_["trgt"]

# If any of the prices in the window deviate more than
# the volatility level for that period then label them as ups/downs (pt/sl)
# otherwise label them as zeros (NANS)

for loc, vertical_barrier in events_["t1"].fillna(close.index[-1]).iteritems():
    df0 = close[loc:vertical_barrier]  # path prices
    df0 = (df0 / close[loc] - 1) * events_.at[loc, "side"]  # path returns
    out.loc[loc, "sl"] = df0[df0 < stop_loss[loc]].index.min()  # earliest stop loss
    out.loc[loc, "pt"] = df0[
        df0 > profit_taking[loc]
    ].index.min()  # earliest profit taking





# Check for duplicates
duplicates_index = df0.index.duplicated(keep='first')
unique, counts = np.unique(duplicates_index, return_counts=True)

duplicates_index_ = sampled_idx.duplicated(keep='first')
unique, counts = np.unique(duplicates_index_, return_counts=True)



# Gettig counts of Y labels
unique_train, counts_train = np.unique(Y_train, return_counts=True)
print(unique_train)
print(counts_train)
unique_val, counts_val = np.unique(Y_val, return_counts=True)
print(unique_val)
print(counts_val)
unique_test, counts_test = np.unique(Y_test, return_counts=True)
print(unique_test)
print(counts_test)

minus_one_count = 0
zero_count = 0
one_count = 0
for row in Y_train:
    if row[0] == 1:
        minus_one_count = minus_one_count + 1
    if row[1] == 1:
        zero_count = zero_count + 1
    if row[2] == 1:
        one_count = one_count + 1

print(minus_one_count)
print(zero_count)
print(one_count)


# model.compile(loss=loss_function, optimizer="adam", metrics=["accuracy"])


# history = model.fit(
#     train,
#     Y_train,
#     epochs=1,
#     batch_size=16,
#     verbose=True,
#     validation_data=(feature_val, Y_val),
#     callbacks=[checkpointer, es, tensorboard],
#     class_weight=None,
# )


# model = keras.models.Sequential(
#     [
#         keras.layers.GRU(20, return_sequences=True, input_shape=[100, 1]),
#         keras.layers.GRU(20, return_sequences=True),
#         keras.layers.GRU(20),
#         keras.layers.Dense(3),
#         keras.layers.Activation(activation="softmax"),
#     ]
# )

# # LSTM
# model = keras.models.Sequential(
#     [
#         keras.layers.LSTM(128, return_sequences=True, input_shape=[100, 1]),
#         keras.layers.LSTM(256, return_sequences=True),
#         keras.layers.LSTM(256, return_sequences=True),
#         keras.layers.LSTM(128),
#         keras.layers.Dense(3),
#         keras.layers.Activation(activation="softmax"),
#     ]
# )


# Y_train = lbr.fit_transform((Y_train))
# Y_val = lbr.fit_transform((Y_val))
# Y_test = lbr.fit_transform((Y_test))

# if stage == 2:
#     fake_x_train, fake_x_test, Y_train, Y_test = train_test_split(
#         P_all_classes, y, test_size=0.2, random_state=1
#     )
#     fake_x_train, fake_x_val, Y_train, Y_val = train_test_split(
#         fake_x_train, Y_train, test_size=0.2, random_state=1
#     )


# X_train, X_test, Y_train, Y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=1
    # )
    # X_train, X_val, Y_train, Y_val = train_test_split(
    #     X_train, Y_train, test_size=0.2, random_state=1
    # )

    # X_train = np.array([np.array(x[:]) for x in X_train]).reshape(
    #     (len(X_train), parameters["WINDOW_LONG"], 1)
    # )
    # X_val = np.array([np.array(x[:]) for x in X_val]).reshape(
    #     (len(X_val), parameters["WINDOW_LONG"], 1)
    # )
    # X_test = np.array([np.array(x[:]) for x in X_test]).reshape(
    #     (len(X_test), parameters["WINDOW_LONG"], 1)
    # )

    # grid = GridSearchCV(estimator=model, param_grid=param_grid)
# grid_result = grid.fit(X_train, Y_train)
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# for params, mean_score, scores in grid_result.grid_scores_:
#     print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
# print("total time:", time() - start)

normed_close = scaler.fit_transform(data[["close"]])
    data["normed_close"] = normed_close


# # Binarize without Oversampling
# lbr = LabelBinarizer()
# y = lbr.fit_transform((y))

# Binarie with Oversampling
# sm = SMOTE()
# sm = ADASYN()  # adds in randomness on top of smote
# X_train = X_train.reshape(len(X_train), len(X_train[0]))
# Y_train = np.asarray(Y_train)
# X_train, Y_train = sm.fit_sample(X_train, Y_train)
# unique, counts = np.unique(Y_val, return_counts=True)

# lbr = LabelBinarizer()
# Y_train = lbr.fit_transform((Y_train))
# X_train = np.array(X_train).reshape((len(X_train), WINDOW_SHAPE, 1))

# Y_val = np.asarray(Y_val)
# Y_val = lbr.fit_transform((Y_val))
# Y_test = np.asarray(Y_test)
# Y_test = lbr.fit_transform((Y_test))


print("min, max, mean, median")
print(data["window_volatility_level"].min())
print(data["window_volatility_level"].max())
print(data["window_volatility_level"].mean())
print(data["window_volatility_level"].median())

print("mean and median")
print(data["window_volatility_level"].mean())
print(data["window_volatility_level"].median())

lbr = LabelBinarizer()
y_binarized = lbr.fit_transform((y))


## XGBOOST
# dtrain = xgb.DMatrix(X_train, label=Y_train, feature_names=feature_names)
# dtest = xgb.DMatrix(X_test, label=Y_test, feature_names=feature_names)
# # specify parameters via map, definition are same as c++ version
# param = {
#     "max_depth": 2,
#     "eta": 1,
#     "silent": 1,
#     "objective": "multi:softmax",
#     "num_class": 3,
#     "tree_method": "gpu_hist",
# }

# # specify validations set to watch performance
# watchlist = [(dtest, "eval"), (dtrain, "train")]
# num_round = 1000
# num_classes = 3
# bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=50)

# fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# xgb.plot_importance(bst, max_num_features=6, ax=ax, importance_type="gain")



### CATBOOST
train_dataset = Pool(data=X_train, label=Y_train)

eval_dataset = Pool(data=X_test, label=Y_test)


# Initialize CatBoostClassifier
model = CatBoostClassifier(
    iterations=1000, learning_rate=1, depth=2, loss_function="MultiClass"
)

model.fit(train_dataset)

feature_importances = model.get_feature_importance(train_dataset)
feature_names = ["p1", "p2", "p3", "p4", "p5"]
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print("{}: {}".format(name, score))

print(model.get_best_iteration())


# df1 = mp_pandas_obj(
    #     func=make_window,
    #     pd_obj=("molecule", prices_for_window.index),
    #     num_threads=cpus * 2,
    #     data=data,
    #     WINDOW_LONG=parameters["WL"],
    # )

    # X = list(itertools.chain.from_iterable(df1))

    # X = np.array([np.array(x[:]) for x in X]).reshape((len(X), parameters["WL"]))
    # end = time.time()
