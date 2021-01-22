from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC, abstractmethod
import os
import warnings
import shutil

import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import Sequence, GeneratorEnqueuer, OrderedEnqueuer
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Layer, Dense, Lambda, Add, LayerNormalization, Concatenate, Dropout
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError

from tensorflow.python.keras.utils.generic_utils import to_list, CustomObjectScope
from tensorflow.python.keras.utils.data_utils import iter_sequence_infinite
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys #?


class DenseBatchNormalization(Layer):
    """Dense and batch-normalization composite layer."""

    def __init__(self, dense, batchnormalization, activation=None, dropout=None, **kwargs):
        super(DenseBatchNormalization, self).__init__(**kwargs)
        self.dense_1 = dense
        self.activation_1 = activation
        self.dropout_1 = dropout
        self.batchnormalization_1 = batchnormalization

    def call(self, inputs, training=None):
        x = inputs
        x = self.dense_1(x)
        if self.activation_1 is not None:
            x = self.activation_1(x)
        if self.dropout_1 is not None:
            x = self.dropout_1(x)

        outputs = x
        return outputs

    def get_config(self):
        """Get configuration."""
        config = {}
        base_config = super(DenseBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))