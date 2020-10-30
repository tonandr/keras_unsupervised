from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.python.keras.layers import Layer, InputSpec, Dense
from tensorflow.python.keras import activations
import tensorflow.keras.initializers as initializers

from ..backend_ext import tensorflow_backend as Ke


class GraphConvolutionNetwork(Layer):
    """Graph convolution network layer."""

    def __init__(self, n_node, d_out, output_adjacency=False, activation=None, **kwargs):
        # Check exception.
        if isinstance(n_node, int) != True \
                or isinstance(d_out, int) != True \
                or (output_adjacency in [False, True]) != True \
                or n_node < 2 \
                or d_out < 1:
            raise ValueError(f'n_node:{n_node}, d_out:{d_out} or output_adjacency:{output_adjacency} is not valid.')

        self.n_node = n_node
        self.d_out = d_out
        self.output_adjacency = output_adjacency
        self.activation = activations.get(activation)
        super(GraphConvolutionNetwork, self).__init__(**kwargs)

    def build(self, input_shape):
        self.d_in = input_shape[0][-1]

        self.I = tf.eye(self.n_node)
        self.W = self.add_weight(name='gcn_weight'
                                 , shape=(self.d_in, self.d_out)
                                 , initializer='truncated_normal' # Which initializer is optimal?
                                 , trainable=True)

        super(GraphConvolutionNetwork, self).build(input_shape)

    def call(self, inputs):
        X = inputs[0]
        A = inputs[1]

        A_t = A + self.I
        D_t = tf.linalg.diag(tf.pow(K.sum(A_t, axis=2), -0.5))
        A_t = K.batch_dot(K.batch_dot(D_t, A_t), D_t)

        X_p = tf.tensordot(K.batch_dot(A_t, X), self.W, axes=[[-1], [0]])

        if self.activation is not None:
            X_p = self.activation(X_p)

        if self.output_adjacency:
            outputs = [X_p, A]
        else:
            outputs = X_p

        return outputs

    def get_config(self):
        config = {'n_node': self.n_node
            , 'd_out': self.d_out}
        base_config = super(GraphConvolutionNetwork, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))