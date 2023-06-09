import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from time import time
from keras.utils import np_utils
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

nb_classes = 10
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255


Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


def create_model(optimizer="rmsprop", init="glorot_uniform"):
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(
        Activation("relu")
    )  # An "activation" is just a non-linear function applied to the output
    model.add(
        Dropout(0.2)
    )  # Dropout helps protect the model from memorizing or "overfitting" the training data
    model.add(Dense(512, kernel_initializer=init))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(10, kernel_initializer=init))
    model.add(Activation("softmax"))  # This special "softmax" a
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

start = time()
model = KerasClassifier(build_fn=create_model)
optimizers = ["rmsprop", "adam"]
init = ["glorot_uniform", "normal", "uniform"]
epochs = np.array([50, 100, 150])
batches = np.array([5, 10, 20])
param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_train, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
print("total time:", time() - start)

