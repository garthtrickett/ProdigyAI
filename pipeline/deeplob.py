# load packages
import pandas as pd
import pickle
import numpy as np
import tensorflow
import tensorflow as tf
import os.path
from os import path

from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy

### These lines are related to mixed precision
# tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
# K.clear_session()
# config = tf.compat.v1.ConfigProto()
# jit_level = tf.compat.v1.OptimizerOptions.ON_1
# config.graph_options.optimizer_options.global_jit_level = jit_level
# sess = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(sess)

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (
    Flatten,
    Dense,
    Dropout,
    Activation,
    Input,
    Reshape,
    Conv2D,
    MaxPooling2D,
    LeakyReLU,
    concatenate,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.compat.v1.keras.backend import set_session
import matplotlib.pyplot as plt
import math
from numba import njit, prange
import yaml

from tensorflow.keras.callbacks import ModelCheckpoint

# from tensorflow.keras.mixed_precision import experimental as mixed_precision
import h5py
import wandb
from wandb.keras import WandbCallback

# check if using gpu
gpus = tf.config.list_physical_devices()
any_gpus = [s for s in gpus if "GPU" in s[0]]

# set random seeds
np.random.seed(1)
tf.random.set_seed(2)

import sys
import os
import argparse

## Path adjustment
sys.path.append("..")
cwd = os.getcwd()
from pathlib import Path
home = str(Path.home())
sys.path.append(home + "/ProdigyAI")

import os
os.environ["TF_KERAS"] = '1'
from third_party_libraries.CLR.clr_callbacks import CyclicLR
from third_party_libraries.lr_finder import LRFinder
from third_party_libraries.cosine_annealing import CosineAnnealingScheduler

# Sorting out whether we are using the ipython kernel or not
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
## FINISHED dealing with ipython kernel check and argparse's

# Init wandb
yaml_path = path_adjust + "yaml/deeplob.yaml"
with open(yaml_path) as file:
    yaml_dict = yaml.load(file, Loader=yaml.FullLoader)

config_dictionary = dict(yaml=yaml_path, params=yaml_dict)

## Check if there is a run in progress
try:
    with open(path_adjust + "temp/deeplob_run_in_progress.txt",
              "r") as text_file:
        stored_id = text_file.read()
    resume = True
except:
    resume = False

## Check whether manual resume was set
if resuming == "resuming":
    resume = "allow"
    wandb_id = stored_id
else:
    wandb_id = wandb.util.generate_id()
    resume = "allow"

wandb.init(
    dir="/home/" + user + "/ProdigyAI/",
    project="prodigyai",
    config=config_dictionary,
    resume=resume,
    entity=user,
    id=wandb_id,
)

window_length = wandb.config["params"]["window_length"]["value"]
num_features = wandb.config["params"]["num_features"]["value"]
epochs = wandb.config["params"]["epochs"]["value"]
batch_size = wandb.config["params"]["batch_size"]["value"]
number_of_lstm = wandb.config["params"]["number_of_lstm"]["value"]

# limit gpu usage for keras
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
import tensorflow.python.keras.backend as K

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

from pathlib import Path
import h5py
home = str(Path.home())

### Load the data
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
y_train = h5f["y_train"][:].astype(np.int8)
y_val = h5f["y_val"][:].astype(np.int8)
y_test = h5f["y_test"][:].astype(np.int8)
h5f.close()

### Limit the amount of rows for testing
head = wandb.config["params"]["head"]["value"]
window_length = len(input_features_normalized_train[0]) - len(y_train)
if head > 0:
    y_train = y_train[:head]
    y_train = y_val[:head]
    prices_for_window_index_array_train = prices_for_window_index_array_train[:
                                                                              head]
    prices_for_window_index_array_val = prices_for_window_index_array_val[:
                                                                          head]
    input_features_normalized_train = input_features_normalized_train[:, :
                                                                      head +
                                                                      window_length]
    input_features_normalized_val = input_features_normalized_val[:, :head +
                                                                  window_length]

## For the softmax and generator
num_classes = y_train[0].shape[0]


@njit
def generate_x_numba(batch_size, dim, list_IDs_temp,
                     input_features_normalized):
    X = np.empty((batch_size, *dim))
    for i, ID in enumerate(list_IDs_temp):
        for k in prange(dim[0]):
            for l in prange(len(input_features_normalized)):
                X[i][k][l] = input_features_normalized[l][ID + k]

    # X[i, ] = self.input_features_normalized[ID+self.window_length, ]
    return X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))


@njit
def generate_y_numba(batch_size, num_classes, list_IDs_temp, y_data):
    y = np.empty((batch_size, num_classes))
    for i, ID in enumerate(list_IDs_temp):

        y[i, :] = y_data[ID]

    return y


class DataGenerator(tf.compat.v2.keras.utils.Sequence):
    def __init__(
        self,
        checkpoint_path,
        prices_for_window_index_array,
        input_features_normalized,
        y_data,
        batch_size,
        dim,
        num_classes,
        to_fit,
        shuffle=True,
    ):
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
            if np.isfinite(y).all() == False:
                print("nans found exiting")
                sys.exit()
            return X, y.astype(np.int8)
        else:
            return X

    def on_epoch_end(self):

        wandb.save(os.path.join(wandb.run.dir, "cp-*"))
        self.indexes = np.arange(len(self.prices_for_window_index_array))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_x(self, list_IDs_temp):

        dim = self.dim
        batch_size = self.batch_size
        input_features_normalized = self.input_features_normalized

        X = generate_x_numba(batch_size, dim, list_IDs_temp,
                             input_features_normalized)

        return X

    def _generate_y(self, list_IDs_temp):

        batch_size = self.batch_size
        num_classes = self.num_classes
        y_data = self.y_data

        y = generate_y_numba(batch_size, num_classes, list_IDs_temp, y_data)

        return y


### Model Architecture
def create_deeplob(T, NF, number_of_lstm, num_classes):
    input_lmd = Input(shape=(T, NF, 1))

    # build the convolutional block
    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(input_lmd)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding="same")(conv_first1)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding="same")(conv_first1)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding="same")(conv_first1)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding="same")(conv_first1)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 10))(conv_first1)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding="same")(conv_first1)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding="same")(conv_first1)
    conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)

    # build the inception module
    convsecond_1 = Conv2D(64, (1, 1), padding="same")(conv_first1)
    convsecond_1 = LeakyReLU(alpha=0.01)(convsecond_1)
    convsecond_1 = Conv2D(64, (3, 1), padding="same")(convsecond_1)
    convsecond_1 = LeakyReLU(alpha=0.01)(convsecond_1)

    convsecond_2 = Conv2D(64, (1, 1), padding="same")(conv_first1)
    convsecond_2 = LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = Conv2D(64, (5, 1), padding="same")(convsecond_2)
    convsecond_2 = LeakyReLU(alpha=0.01)(convsecond_2)

    convsecond_3 = MaxPooling2D((3, 1), strides=(1, 1),
                                padding="same")(conv_first1)
    convsecond_3 = Conv2D(64, (1, 1), padding="same")(convsecond_3)
    convsecond_3 = LeakyReLU(alpha=0.01)(convsecond_3)

    convsecond_output = concatenate([convsecond_1, convsecond_2, convsecond_3],
                                    axis=3)

    # use the MC dropout here
    conv_reshape = Reshape(
        (int(convsecond_output.shape[1]),
         int(convsecond_output.shape[3])))(convsecond_output)

    conv_lstm = layers.LSTM(number_of_lstm)(conv_reshape)

    # build the output layer
    out = Dense(num_classes, activation="softmax", dtype="float32")(conv_lstm)
    model = Model(inputs=input_lmd, outputs=out)
    opt = wandb.config["params"]["opt"]["value"]
    if opt == "adam":
        opt = Adam(lr=1e-02, beta_1=0.9, beta_2=0.999, epsilon=1)
    elif opt == "sgd":
        opt = SGD()
    # opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
    #     adam, "dynamic")

    model.compile(optimizer=opt,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model


##

## Deal with the checkpoints
checkpoint_path = wandb.run.dir + "/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
)

# Model Params
dim = (window_length, num_features)
to_fit = True
loss_function = categorical_crossentropy
no_epochs = 1

train_generator = DataGenerator(
    checkpoint_path,
    prices_for_window_index_array_train,
    input_features_normalized_train,
    y_train,
    batch_size,
    dim,
    num_classes,
    to_fit,
    shuffle=True,
)

val_generator = DataGenerator(
    checkpoint_path,
    prices_for_window_index_array_val,
    input_features_normalized_val,
    y_val,
    batch_size,
    dim,
    num_classes,
    to_fit,
    shuffle=True,
)

## More model params
steps_per_epoch = len(train_generator)
validation_steps = len(val_generator)

## Load and resume model if needed
deeplob = create_deeplob(window_length, num_features, number_of_lstm,
                         num_classes)

if resuming == "resuming":
    try:
        # restore the best model
        K.clear_session()
        from tensorflow.keras.models import model_from_json
        json_path = wandb.restore('model.json',
                                  run_path=user + '/prodigyai/' +
                                  wandb_id).name
        json_file = open(json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        deeplob = model_from_json(loaded_model_json)
        weights_path = wandb.restore('weights.h5',
                                     run_path=user + '/prodigyai/' +
                                     wandb_id).name
        deeplob.load_weights(weights_path)
        if opt == "adam":
            opt = Adam(lr=1e-02, beta_1=0.9, beta_2=0.999, epsilon=1)
        elif opt == "sgd":
            opt = SGD()
        deeplob.compile(optimizer=opt,
                        loss="categorical_crossentropy",
                        metrics=["accuracy"])
        # deeplob = tensorflow.keras.models.load_model(
        #     wandb.restore('my_model.h5',
        #                   run_path=user + '/prodigyai/' + wandb_id).name)

    except:
        print("no prior epochs completed")

with open(path_adjust + "temp/deeplob_run_in_progress.txt", "w+") as text_file:
    text_file.write(wandb_id)

## Find Learning rates
use_lr_finder = wandb.config["params"]["use_lr_finder"]["value"]
if use_lr_finder is True:
    lr_finder = LRFinder(wandb=wandb,
                         max_steps=steps_per_epoch,
                         start_lr=1e-03,
                         end_lr=5)

    deeplob.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=1,
        validation_data=val_generator,
        callbacks=[
            cp_callback, lr_finder,
            WandbCallback(save_model=True, monitor="val_loss", mode="min")
        ],
    )

    lr_finder.plot()

run_training = wandb.config["params"]["run_training"]["value"]

if run_training is True:
    use_clr = wandb.config["params"]["use_clr"]["value"]

    if use_clr is True:
        # Set CLR options
        clr_step_size = int(
            4 * (len(prices_for_window_index_array_train) / batch_size))
        base_lr = 1e-02
        max_lr = 1e-01
        mode = "triangular"
        # Define the callback
        schedule = CyclicLR(base_lr=base_lr,
                            max_lr=max_lr,
                            step_size=clr_step_size,
                            mode=mode)

    use_sgdr = wandb.config["params"]["use_sgdr"]["value"]

    if use_sgdr is True:
        schedule = CosineAnnealingScheduler(T_max=100,
                                            eta_max=1e-1,
                                            eta_min=1e-2)

    for i in range(epochs):
        try:
            callbacks = [
                cp_callback, schedule,
                WandbCallback(save_model=True, monitor="val_loss", mode="min")
            ]
        except:
            print("no schedule callback")
            callbacks = [
                cp_callback,
                WandbCallback(save_model=True, monitor="val_loss", mode="min")
            ]
        deeplob.fit(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps,
                    epochs=1,
                    validation_data=val_generator,
                    callbacks=callbacks)
        # save the model
        weights_path = wandb.run.dir + "/weights.h5"
        model_path = wandb.run.dir + "/model.json"
        deeplob.save_weights(weights_path)
        wandb.save(os.path.join(wandb.run.dir, "weights.h5"))
        wandb.save(os.path.join(wandb.run.dir, "model.json"))
        model_json = deeplob.to_json()
        with open(model_path, "w") as json_file:
            json_file.write(model_json)

    finished_weights_path = path_adjust + "temp/cp_end.ckpt"
    deeplob.save_weights(finished_weights_path)

    with open(path_adjust + "temp/run_finished.txt", "w+") as text_file:
        text_file.write(wandb.run.id)

    os.remove(path_adjust + "temp/deeplob_run_in_progress.txt")

    print("script finished")
