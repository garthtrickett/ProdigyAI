print("script initiated")

import argparse
import os
import sys

import imblearn
from imblearn.over_sampling import SMOTE
import tensorflow
import matplotlib.pyplot as plt

sys.path.append("..")
cwd = os.getcwd()
from pathlib import Path

home = str(Path.home())
sys.path.append(home + "/ProdigyAI")

from numba import njit, prange

import math

import numpy as np
from third_party_libraries.TABL import Models
from third_party_libraries.keras_lr_finder.lr_finder import LRFinder
from third_party_libraries.CLR.clr_callbacks import CyclicLR

from sklearn.utils.class_weight import compute_class_weight

import keras
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
# tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras import backend as K
K.clear_session()
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)
# config = tf.compat.v1.ConfigProto()
# jit_level = tf.compat.v1.OptimizerOptions.ON_1
# config.graph_options.optimizer_options.global_jit_level = jit_level
# sess = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(sess)

# Init wandb
import wandb
from wandb.keras import WandbCallback
from third_party_libraries.LRFT.keras_callbacks import LRFinder
import yaml

import getpass
user = getpass.getuser()

# Sorting out whether we are using the ipython kernel or not
try:
    resuming = "NA"
    get_ipython()
    check_if_ipython = True
    path_adjust = "../"
    import getpass
    user = getpass.getuser()

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
    parser.add_argument("-u",
                        "--user",
                        type=str,
                        help="Stage of Preprocesssing")
    parser.add_argument("-f",
                        "--resuming",
                        type=str,
                        help="Is this a continuation of preempted instance?")
    parser.add_argument("-m",
                        "--model",
                        type=str,
                        help="one_model or two_model")
    args = parser.parse_args()
    if args.user != None:
        user = args.user
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

yaml_path = path_adjust + "yaml/tabl.yaml"
with open(yaml_path) as file:
    yaml_dict = yaml.load(file, Loader=yaml.FullLoader)
config_dictionary = dict(yaml=yaml_path, params=yaml_dict)

try:
    with open(path_adjust + "temp/tabl_run_in_progress.txt", "r") as text_file:
        stored_id = text_file.read()
    resume = True
except:
    resume = False

if resuming == "resuming":
    resume = "allow"
    wandb_id = stored_id
else:
    wandb_id = wandb.util.generate_id()
    resume = "allow"

wandb.init(dir="/home/" + user + "/ProdigyAI/",
           project="prodigyai",
           config=config_dictionary,
           resume=resume,
           entity=user,
           id=wandb_id)

window_length = wandb.config['params']['window_length']['value']
num_features = wandb.config['params']['num_features']['value']
epochs = wandb.config['params']['epochs']['value']
batch_size = wandb.config['params']['batch_size']['value']
dropout = wandb.config['params']['dropout']['value']

## PRODIGY AI HOCKUS POCKUS START
from pathlib import Path
import h5py

home = str(Path.home())

file_name = wandb.config["params"]["dataset"]["value"]
path = home + "/ProdigyAI/data/preprocessed/" + file_name
h5f = h5py.File(path, "r")
prices_for_window_index_array_train = h5f[
    "prices_for_window_index_array_train"][:]
prices_for_window_index_array_val = h5f["prices_for_window_index_array_val"][:]
prices_for_window_index_array_test = h5f[
    "prices_for_window_index_array_test"][:]
input_features_normalized_train = h5f["input_features_normalized_train"][:]
input_features_normalized_val = h5f["input_features_normalized_val"][:]
input_features_normalized_test = h5f["input_features_normalized_test"][:]
y_train = h5f["y_train"][:]
y_val = h5f["y_val"][:]
y_test = h5f["y_test"][:]
h5f.close()
## PRODIGY AI HOCKUS POCKUS END

# Class Weights
y_train_class_one_count = np.unique(y_train[:, 0], return_counts=True)[1][1]
y_train_class_two_count = np.unique(y_train[:, 1], return_counts=True)[1][1]
y_train_class_three_count = np.unique(y_train[:, 2], return_counts=True)[1][1]
print(y_train_class_one_count)
print(y_train_class_two_count)
print(y_train_class_three_count)

y_train_integers = np.argmax(y_train, axis=1)
class_weights_train = compute_class_weight('balanced',
                                           np.unique(y_train_integers),
                                           y_train_integers)
class_weights_train_dict = dict(enumerate(class_weights_train))

import pdb
pdb.set_trace()

num_classes = y_train[0].shape[0]


@njit
def generate_x_numba(batch_size, dim, list_IDs_temp,
                     input_features_normalized):
    X = np.empty((batch_size, dim[1], dim[0]))
    for i, ID in enumerate(list_IDs_temp):
        for k in prange(len(input_features_normalized)):
            for l in prange(dim[0]):
                X[i][k][l] = input_features_normalized[k][ID + l]

    # X[i, ] = self.input_features_normalized[ID+self.window_length, ]
    return X


@njit
def generate_y_numba(batch_size, num_classes, list_IDs_temp, y_data):

    y = np.empty((batch_size, num_classes))

    for i, ID in enumerate(list_IDs_temp):

        y[i, :] = y_data[ID]

    return y


class DataGenerator(tf.compat.v2.keras.utils.Sequence):
    def __init__(self,
                 checkpoint_path,
                 prices_for_window_index_array,
                 input_features_normalized,
                 y_data,
                 batch_size,
                 dim,
                 num_classes,
                 to_fit,
                 shuffle=True):
        self.batch_count = 0
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.prices_for_window_index_array = prices_for_window_index_array
        self.input_features_normalized = input_features_normalized
        self.labels = y_data
        self.y_data = y_data
        self.to_fit = to_fit
        self.num_classes = num_classes
        self.dim = dim
        self.shuffle = shuffle
        self.n = 0
        self.list_IDs = np.arange(len(self.prices_for_window_index_array))
        self.on_epoch_end()

    def __next__(self):
        # Get one batch of data
        data = self.__getitem__(self.n)
        # Batch index
        self.n += 1

        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.on_epoch_end
            self.n = 0

        return data

    def __len__(self):
        # Return the number of batches of the dataset
        return math.ceil(len(self.indexes) / self.batch_size)

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = self._generate_x(list_IDs_temp)
        if np.isfinite(X).all() == False:
            print("nans found exiting")
            sys.exit()

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)

            #instead of concatenating could append to file like binance_api

            if self.batch_count == 0:
                self.y_batch_total = y
            else:
                self.y_batch_total = np.concatenate([self.y_batch_total, y])
            self.batch_count = self.batch_count + 1
            if np.isfinite(y).all() == False:
                print("nans found exiting")
                sys.exit()

            return X, y.astype(np.int8)
        else:
            return X

    def on_epoch_end(self):

        wandb.save(self.checkpoint_path)

        self.indexes = np.arange(len(self.prices_for_window_index_array))

        if self.batch_count > 0:
            h5f = h5py.File(path_adjust + "temp/y_batch_total.h5", "w")
            h5f.create_dataset("y_batch_total", data=self.y_batch_total)
            h5f.close()

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_x(self, list_IDs_temp):

        dim = self.dim
        batch_size = self.batch_size
        input_features_normalized = self.input_features_normalized

        X = generate_x_numba(batch_size, dim, list_IDs_temp,
                             input_features_normalized)
        # return X.astype('float16')
        return X

    def _generate_y(self, list_IDs_temp):

        batch_size = self.batch_size
        num_classes = self.num_classes
        y_data = self.y_data

        y = generate_y_numba(batch_size, num_classes, list_IDs_temp, y_data)

        return y


# Model configuration
dim = (window_length, num_features)
to_fit = True
batch_size = 256
loss_function = categorical_crossentropy
no_epochs = 1
start_lr = 0.0001
end_lr = 1
moving_average = 20

checkpoint_path = os.path.join(wandb.run.dir, "cp.ckpt")

# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                              save_weights_only=True,
                              verbose=1)

train_generator = DataGenerator(checkpoint_path,
                                prices_for_window_index_array_train,
                                input_features_normalized_train,
                                y_train,
                                batch_size,
                                dim,
                                num_classes,
                                to_fit,
                                shuffle=True)

val_generator = DataGenerator(checkpoint_path,
                              prices_for_window_index_array_val,
                              input_features_normalized_val,
                              y_val,
                              batch_size,
                              dim,
                              num_classes,
                              to_fit,
                              shuffle=True)

steps_per_epoch = len(train_generator)
validation_steps = len(val_generator)

# for X_batch, y_batch in DataGenerator(prices_for_window_index_array_train,
#                                       input_features_normalized_train,
#                                       y_train,
#                                       batch_size,
#                                       dim,
#                                       num_classes,
#                                       to_fit,
#                                       shuffle=True):
#     import pdb
#     pdb.set_trace()

# 1 hidden layer network with input: 40x10, hidden 120x5, output 3x1

template = [[num_features, window_length], [60, 10], [120, 5],
            [num_classes, 1]]

# get Bilinear model
projection_regularizer = None
projection_constraint = keras.constraints.max_norm(3.0, axis=0)
attention_regularizer = None
attention_constraint = keras.constraints.max_norm(5.0, axis=1)

# Determine tests you want to perform
tests = [
    ("sgd", "SGD optimizer"),
    ("adam", "Adam optimizer"),
    ("rmsprop", "RMS Prop optimizer"),
]

# Set containers for tests
test_learning_rates = []
test_losses = []
test_loss_changes = []
labels = []

model = Models.TABL(
    template,
    dropout,
    projection_regularizer,
    projection_constraint,
    attention_regularizer,
    attention_constraint,
)

wandb_callback = WandbCallback(save_model=True, monitor="loss")

# # Perform each test
# for test_optimizer, label in tests:

#     # Compile the model
#     model.compile(loss=loss_function,
#                   optimizer=test_optimizer,
#                   metrics=["accuracy"])

#     # Instantiate the Learning Rate Range Test / LR Finder
#     lr_finder = LRFinder(model)

#     # Perform the Learning Rate Range Test
#     outputs = lr_finder.find_generator(train_generator,
#                                        start_lr=start_lr,
#                                        end_lr=end_lr,
#                                        epochs=no_epochs,
#                                        cp_callback=cp_callback,
#                                        wandb_callback=wandb_callback)

#     # Get values
#     learning_rates = lr_finder.lrs
#     losses = lr_finder.losses
#     loss_changes = []

#     # Compute smoothed loss changes
#     # Inspired by Keras LR Finder: https://github.com/surmenok/keras_lr_finder/blob/master/keras_lr_finder/lr_finder.py
#     for i in range(moving_average, len(learning_rates)):
#         loss_changes.append(
#             (losses[i] - losses[i - moving_average]) / moving_average)

#     # Append values to container
#     test_learning_rates.append(learning_rates)
#     test_losses.append(losses)
#     test_loss_changes.append(loss_changes)
#     labels.append(label)

# # Generate plot for Loss Deltas
# for i in range(0, len(test_learning_rates)):
#     plt.plot(test_learning_rates[i][moving_average:],
#              test_loss_changes[i],
#              label=labels[i])
# plt.xscale("log")
# plt.legend(loc="upper left")
# plt.ylabel("loss delta")
# plt.xlabel("learning rate (log scale)")
# plt.title(
#     "Results for Learning Rate Range Test / Loss Deltas for Learning Rate")
# plt.show()
# plt.savefig(wandb.run.dir + 'plot1.png')
# wandb.save(wandb.run.dir + 'plot1.png')

# # Generate plot for Loss Values
# for i in range(0, len(test_learning_rates)):
#     plt.plot(test_learning_rates[i], test_losses[i], label=labels[i])
# plt.xscale("log")
# plt.legend(loc="upper left")
# plt.ylabel("loss")
# plt.xlabel("learning rate (log scale)")
# plt.title(
#     "Results for Learning Rate Range Test / Loss Values for Learning Rate")
# plt.show()
# plt.savefig(wandb.run.dir + 'plot2.png')
# wandb.save(wandb.run.dir + 'plot2.png')

# Configuration settings for LR finder
start_lr = 1e-4
end_lr = 1e0
no_epochs = 1

##
## LR Finder specific code
##
opt = keras.optimizers.Adam(0.01)
# opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score


@njit
def get_three_class_one_hot_from_three_class_probabilities(y_val_pred):
    y_val_pred_classes = np.zeros((len(y_val_pred), 3))
    y_val_pred_classes[:] = np.nan

    for i in range(len(y_val_pred)):
        if np.argmax(y_val_pred[0], axis=0) == 0:
            y_val_pred_classes[i] = [1, 0, 0]
        elif np.argmax(y_val_pred[0], axis=0) == 1:
            y_val_pred_classes[i] = [0, 1, 0]
        elif np.argmax(y_val_pred[0], axis=0) == 2:
            y_val_pred_classes[i] = [0, 0, 1]
    return y_val_pred_classes


class CustomMetrics(keras.callbacks.Callback):
    def __init__(self, train_generator, val_generator, validation_steps,
                 y_train, y_val):
        self.val_generator = val_generator
        self.validation_steps = validation_steps
        self.train_generator = train_generator
        self.y_train = y_train
        self.y_val = y_val
        self.batch_count = 0

    def on_epoch_end(self, batch, logs={}):
        y_val_pred = self.model.predict(self.val_generator,
                                        steps=self.validation_steps)
        h5f = h5py.File(path_adjust + "temp/y_batch_total.h5", "r")
        y_batch_total = h5f["y_batch_total"][:]
        h5f.close()

        roc_val = roc_auc_score(y_batch_total, y_val_pred)

        y_val_pred_classes = get_three_class_one_hot_from_three_class_probabilities(
            y_val_pred)

        import pdb
        pdb.set_trace()

        accuracy_score(y_batch_total, y_val_pred_classes)
        self.scores = {'validation_auc_roc': []}

        return


metrics_callback = CustomMetrics(train_generator, val_generator,
                                 validation_steps, y_train, y_val)

metrics = [
    "accuracy",
    # tensorflow.keras.metrics.TruePositives(name='tp'),
    # tensorflow.keras.metrics.FalsePositives(name='fp'),
    # tensorflow.keras.metrics.TrueNegatives(name='tn'),
    # tensorflow.keras.metrics.FalseNegatives(name='fn'),
    # tensorflow.keras.metrics.BinaryAccuracy(name='accuracy'),
    # tensorflow.keras.metrics.Precision(name='precision'),
    # tensorflow.keras.metrics.Recall(name='recall'),
    tensorflow.keras.metrics.AUC(name='auc'),
]

# Compile the model
model.compile(loss=loss_function, optimizer=opt, metrics=metrics)
# Define LR finder callback
lr_finder = LRFinder(min_lr=start_lr, max_lr=end_lr, wandb=wandb)

# Perform LR finder
# model.fit(
#     train_generator,
#     steps_per_epoch=steps_per_epoch,
#     validation_steps=validation_steps,
#     validation_data=val_generator,
#     callbacks=[lr_finder, cp_callback, wandb_callback],
#     epochs=no_epochs,
# )

print("learning rate found running cyclical learning")

# ### cyclical learning rate

# Set CLR options
clr_step_size = int(4 *
                    (len(prices_for_window_index_array_train) / batch_size))
base_lr = 1e-4
max_lr = 1e-2
mode = "triangular"
# Define the callback
clr = CyclicLR(base_lr=base_lr,
               max_lr=max_lr,
               step_size=clr_step_size,
               mode=mode)

model = Models.TABL(
    template,
    dropout,
    projection_regularizer,
    projection_constraint,
    attention_regularizer,
    attention_constraint,
)
model.summary()

opt = keras.optimizers.Adam(0.01)
# opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")

# Compile the model
model.compile(loss=loss_function, optimizer=opt, metrics=metrics)

if wandb.run.resumed:
    weights = wandb.restore("cp.ckpt",
                            run_path=user + "/prodigyai/" + wandb_id)
    model.load_weights(weights.name)

# create class weight
# class_weight = {0: 1e6 / 300.0, 1: 1e6 / 400.0, 2: 1e6 / 300.0}

with open(path_adjust + "temp/tabl_run_in_progress.txt", "w+") as text_file:
    text_file.write(wandb_id)

# example sata training
model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=epochs,
    validation_data=val_generator,
    class_weight=class_weights_train_dict,
    callbacks=[
        # clr,
        cp_callback,
        wandb_callback,
        # metrics_callback
    ])

model.save(os.path.join(wandb.run.dir, "model.h5"))

with open(path_adjust + "temp/run_finished.txt", "w+") as text_file:
    text_file.write(wandb_id)

os.remove(path_adjust + "temp/tabl_run_in_progress.txt")

# no class weight

## lob FI-2010 DATA PREPERATION
# The first 40 columns of the FI-2010 dataset are 10 levels ask and bid information
# for a limit order book and we only use these 40 features in our network.
# The last 5 columns of the FI-2010 dataset are the labels with different prediction horizons.

# def prepare_x(data):
#     df1 = data[:40, :].T
#     return np.array(df1)

# def get_label(data):
#     lob = data[-5:, :].T
#     return lob

# def data_classification(X, Y, T):
#     [N, D] = X.shape
#     df = np.array(X)

#     dY = np.array(Y)

#     dataY = dY[T - 1 : N]

#     dataX = np.zeros((N - T + 1, T, D))
#     for i in range(T, N + 1):
#         dataX[i - T] = df[i - T : i, :]

#     return dataX.reshape(dataX.shape + (1,)), dataY

# # please change the data_path to your local path
# data_path = (
#     "../third_party_libraries/gam_rhn/95-FI2010/data/BenchmarkDatasets/NoAuction"
# )

# dec_train = np.loadtxt(
#     data_path
#     + "/3.NoAuction_DecPre/NoAuction_DecPre_Training/Train_Dst_NoAuction_DecPre_CF_7.txt"
# )
# dec_test1 = np.loadtxt(
#     data_path
#     + "/3.NoAuction_DecPre/NoAuction_DecPre_Testing/Test_Dst_NoAuction_DecPre_CF_7.txt"
# )
# dec_test2 = np.loadtxt(
#     data_path
#     + "/3.NoAuction_DecPre/NoAuction_DecPre_Testing/Test_Dst_NoAuction_DecPre_CF_8.txt"
# )
# dec_test3 = np.loadtxt(
#     data_path
#     + "/3.NoAuction_DecPre/NoAuction_DecPre_Testing/Test_Dst_NoAuction_DecPre_CF_9.txt"
# )

# dec_test = np.hstack((dec_test1, dec_test2, dec_test3))

# # extract limit order book data from the FI-2010 dataset
# train_lob = prepare_x(dec_train)
# test_lob = prepare_x(dec_test)

# # extract label from the FI-2010 dataset
# train_label = get_label(dec_train)
# test_label = get_label(dec_test)

# # prepare training data. We feed past 100 observations into our algorithms and choose the prediction horizon.
# trainX_CNN, trainY_CNN = data_classification(train_lob, train_label, T=10)
# trainY_CNN = trainY_CNN[:, 3] - 1
# trainY_CNN = np_utils.to_categorical(trainY_CNN, 3)

# # trainX_CNN.shape (254651, 100, 40, 1)

# # prepare test data.
# testX_CNN, testY_CNN = data_classification(test_lob, test_label, T=10)
# testY_CNN = testY_CNN[:, 3] - 1
# testY_CNN = np_utils.to_categorical(testY_CNN, 3)

# trainX_CNN = np.swapaxes(trainX_CNN, 1, 2)
# trainX_CNN = trainX_CNN.reshape(
#     trainX_CNN.shape[0], trainX_CNN.shape[1], trainX_CNN.shape[2]
# )

# testX_CNN = testX_CNN = np.swapaxes(testX_CNN, 1, 2)
# testX_CNN = testX_CNN.reshape(
#     testX_CNN.shape[0], testX_CNN.shape[1], testX_CNN.shape[2]
# )

# path = path_adjust + "data/lob_2010/train_and_test_tabl.h5"
# h5f = h5py.File(path, "w")
# h5f.create_dataset("trainX_CNN", data=trainX_CNN)
# h5f.create_dataset("trainY_CNN", data=trainY_CNN)
# h5f.create_dataset("testX_CNN", data=testX_CNN)
# h5f.create_dataset("testY_CNN", data=testY_CNN)
# h5f.close()

# fi-02010 data

# path = path_adjust + "data/lob_2010/train_and_test_tabl.h5"
# h5f = h5py.File(path, "r")
# trainX_CNN = h5f["trainX_CNN"][:]
# trainY_CNN = h5f["trainY_CNN"][:]
# testX_CNN = h5f["testX_CNN"][:]
# testY_CNN = h5f["testY_CNN"][:]
# h5f.close()
# Fit data to model
# history = model.fit(trainX_CNN,
#                     trainY_CNN,
#                     batch_size=batch_size,
#                     epochs=20,
#                     callbacks=[clr, cp_callback])

# score = model.evaluate(x=X_test, y=y_test, batch_size=256)

# Save model to wandb

# print(score)

# tabl score on deeplob version of fi-2010 loss: 0.7469 - acc: 0.6760 - val_loss: 0.6772 - val_acc: 0.7248
