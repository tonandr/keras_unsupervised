from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import copy
from six.moves import zip

import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.optimizers import Adam, Optimizer

import tensorflow as tf

# TODO