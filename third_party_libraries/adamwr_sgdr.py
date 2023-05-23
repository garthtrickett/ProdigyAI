"""From built-in optimizer classes.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import copy
from six.moves import zip

from tensorflow import keras

from keras import backend as K
from keras.utils.generic_utils import serialize_keras_object
from keras.utils.generic_utils import deserialize_keras_object
from keras.legacy import interfaces
from keras.callbacks import Callback
from keras.optimizers import Optimizer


class WRScheduler(Callback):
    """Warm restart scheduler for optimizers with decoupled weight decay.
    
    Warm restarts include cosine annealing with periodic restarts
    for both learning rate and weight decay. Normalized weight decay is also included.
    
    # Arguments
        steps_per_epoch: int > 0. The number of training batches per epoch.
        eta_min: float >=0. The minimum of the multiplier.
        eta_max: float >=0. The maximum of the multiplier.
        eta_decay: float >=0. The decay rate of eta_min/eta_max after each restart.
        cycle_length: int > 0. The number of epochs in the first restart cycle.
        cycle_mult_factor: float > 0. The rate to increase the number of epochs 
            in a cycle after each restart.
            
    # Reference
        - [SGDR: Stochastic Gradient Descent with Warm Restarts](http://arxiv.org/abs/1608.03983)
        - [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
    """
    def __init__(self,
                 steps_per_epoch,
                 eta_min=0.0,
                 eta_max=1.0,
                 eta_decay=1.0,
                 cycle_length=10,
                 cycle_mult_factor=1.5):

        super(WRScheduler, self).__init__()

        self.steps_per_epoch = steps_per_epoch

        self.eta_min = eta_min
        self.eta_max = eta_max
        self.eta_decay = eta_decay

        self.steps_since_restart = 0
        self.next_restart = cycle_length

        self.cycle_length = cycle_length
        self.cycle_mult_factor = cycle_mult_factor

    def cal_eta(self):
        '''Calculate eta'''
        fraction_to_restart = self.steps_since_restart / (
            self.steps_per_epoch * self.cycle_length)
        eta = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (
            1.0 + np.cos(fraction_to_restart * np.pi))
        return eta

    def on_train_begin(self, logs={}):
        '''Set the number of training batches of the first restart cycle to steps_per_cycle'''
        K.set_value(self.model.optimizer.steps_per_cycle,
                    self.steps_per_epoch * self.cycle_length)

    def on_train_batch_begin(self, batch, logs={}):
        '''update eta'''
        eta = self.cal_eta()
        K.set_value(self.model.optimizer.eta, eta)
        self.steps_since_restart += 1

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary'''
        if epoch + 1 == self.next_restart:
            self.steps_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length *
                                        self.cycle_mult_factor)
            self.next_restart += self.cycle_length
            self.eta_min *= self.eta_decay
            self.eta_max *= self.eta_decay
            K.set_value(self.model.optimizer.steps_per_cycle,
                        self.steps_per_epoch * self.cycle_length)


class SGDW(Optimizer):
    """Stochastic gradient descent optimizer with decoupled weight decay.
    Includes support for momentum, learning rate decay, Nesterov momentum,
    and warm restarts.
    # Arguments
        learning_rate: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
        weight_decay: float >= 0. Normalized weight decay.
        eta: float >= 0. The multiplier to schedule learning rate and weight decay.
        steps_per_cycle: int > 0. The number of training batches of a restart cycle.
        
    # References
        - [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
    """
    def __init__(self,
                 learning_rate=0.01,
                 momentum=0.,
                 decay=0.,
                 nesterov=False,
                 weight_decay=0.025,
                 eta=1.0,
                 steps_per_cycle=1,
                 **kwargs):
        super(SGDW, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate,
                                            name='learning_rate')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
            self.eta = K.variable(eta, name='eta')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
            self.steps_per_cycle = K.variable(steps_per_cycle,
                                              name='steps_per_cycle')
        self.initial_decay = decay
        self.nesterov = nesterov

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        w_d = self.eta * self.weight_decay / K.sqrt(self.steps_per_cycle)

        learning_rate = self.eta * self.learning_rate
        if self.initial_decay > 0:
            learning_rate = learning_rate * (1. / (1. + self.decay * K.cast(
                self.iterations, K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - learning_rate * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - learning_rate * g - w_d * p
            else:
                new_p = p + v - w_d * p

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'learning_rate': float(K.get_value(self.learning_rate)),
            'momentum': float(K.get_value(self.momentum)),
            'decay': float(K.get_value(self.decay)),
            'nesterov': self.nesterov,
            'weight_decay': float(K.get_value(self.weight_decay)),
            'eta': float(K.get_value(self.eta)),
            'steps_per_cycle': int(K.get_value(self.steps_per_cycle))
        }
        base_config = super(SGDW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay.
    Default parameters follow those provided in the original Adam paper.
    
    # Arguments
        learning_rate: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: float >= 0. Normalized weight decay.
        eta: float >= 0. The multiplier to schedule learning rate and weight decay.
        steps_per_cycle: int > 0. The number of training batches of a restart cycle.
        
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
        - [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
    """
    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=None,
                 decay=0.,
                 weight_decay=0.025,
                 eta=1.0,
                 steps_per_cycle=1,
                 **kwargs):
        super(AdamW, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate,
                                            name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.eta = K.variable(eta, name='eta')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
            self.steps_per_cycle = K.variable(steps_per_cycle,
                                              name='steps_per_cycle')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        w_d = self.eta * self.weight_decay / K.sqrt(self.steps_per_cycle)

        learning_rate = self.eta * self.learning_rate
        if self.initial_decay > 0:
            learning_rate = learning_rate * (1. / (1. + self.decay * K.cast(
                self.iterations, K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1

        learning_rate_t = learning_rate * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                                           (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            p_t = p - learning_rate_t * m_t / (K.sqrt(v_t) +
                                               self.epsilon) - w_d * p

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'learning_rate': float(K.get_value(self.learning_rate)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'weight_decay': float(K.get_value(self.weight_decay)),
            'eta': float(K.get_value(self.eta)),
            'steps_per_cycle': int(K.get_value(self.steps_per_cycle)),
            'epsilon': self.epsilon
        }
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
