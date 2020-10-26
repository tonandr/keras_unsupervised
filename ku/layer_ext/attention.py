from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras.layers import Layer, InputSpec, Dense, Conv2D, Conv3D

# Constants.
SIMILARITY_TYPE_DIFF_ABS = 'diff_abs'
SIMILARITY_TYPE_PLAIN = 'plain'
SIMILARITY_TYPE_SCALED = 'scaled'
SIMILARITY_TYPE_GENERAL = 'general'
SIMILARITY_TYPE_ADDITIVE = 'additive'


class MultiHeadAttention(Layer):
    """Multi-head attention."""

    def __init__(self, num_head, d_output, dropout_rate, similarity_type=SIMILARITY_TYPE_SCALED, **kwargs):
        # Check exception.
        if isinstance(num_head, int) != True \
                        or isinstance(d_output, int) != True \
                        or (similarity_type in [SIMILARITY_TYPE_DIFF_ABS
                                        , SIMILARITY_TYPE_PLAIN
                                        , SIMILARITY_TYPE_SCALED
                                        , SIMILARITY_TYPE_GENERAL
                                        , SIMILARITY_TYPE_ADDITIVE]) != True \
                        or num_head < 1 \
                        or d_output < 1 \
                        or dropout_rate < 0:
            raise ValueError('num_head, d_output, dropout_rate or similarity_type is not valid.')

        self.num_head = num_head
        self.d_output = d_output
        self.dropout_rate = dropout_rate
        self.similarity_type = similarity_type
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('MultiHeadAttention layer should be called '
                             'on four Q, K, V, M inputs')

        # Get dimension of K, V.
        self.d_k = input_shape[1][-1]
        self.d_v = input_shape[2][-1]

        assert self.d_k % self.num_head == 0 and self.d_v % self.num_head == 0

        self.d_k_h = self.d_k // self.num_head
        self.d_v_h = self.d_v // self.num_head

        # Create linear weight tensors of Q, K, V for each head.
        self.W_Q = self.add_weight(name='Q_linear_weight'
                                 , shape=(self.d_k, self.d_k)
                                 , initializer='truncated_normal' # Which initializer is optimal?
                                 , trainable=True)
        self.W_K = self.add_weight(name='K_linear_weight'
                                 , shape=(self.d_k, self.d_k)
                                 , initializer='truncated_normal' # Which initializer is optimal?
                                 , trainable=True)
        self.W_V = self.add_weight(name='V_linear_weight'
                                 , shape=(self.d_v, self.d_v)
                                 , initializer='truncated_normal' # Which initializer is optimal?
                                 , trainable=True)

        # Create weight tensors for similarity calculation.
        if self.similarity_type == SIMILARITY_TYPE_GENERAL:
            self.W_gen_S = self.add_weight(name='general_similarity_weight'
                                 , shape=(self.d_k_h, self.d_k_h)
                                 , initializer='truncated_normal' # Which initializer is optimal?
                                 , trainable=True)
        elif self.similarity_type == SIMILARITY_TYPE_ADDITIVE:
            self.W_add_S_Q = self.add_weight(name='Q_additive_similarity_weight'
                                 , shape=(self.d_k_h, self.d_k_h)
                                 , initializer='truncated_normal' # Which initializer is optimal?
                                 , trainable=True)
            self.W_add_S_K = self.add_weight(name='K_additive_similarity_weight'
                                 , shape=(self.d_k_h, self.d_k_h)
                                 , initializer='truncated_normal' # Which initializer is optimal?
                                 , trainable=True)

        # Create the multi-head weight.
        self.W_multi_head = self.add_weight(name='multi_head_weight'
                                 , shape=(self.d_v, self.d_output)
                                 , initializer='truncated_normal' # Which initializer is optimal?
                                 , trainable=True)

        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs, training=None):
        # Check exception.
        x = inputs
        if isinstance(x, list) != True or len(x) != 4:
            raise ValueError('Inputs must be a list of three Q, K, V, M tensors.')
        assert backend.int_shape(x[0]) == backend.int_shape(x[1])  # ?

        Q = inputs[0]
        K = inputs[1]
        V = inputs[2]
        M = inputs[3]

        batch_size = tf.shape(K)[0]

        # Do self-attention according to the number of heads and a similarity type.
        # Do linear transformation.
        Q_l = tf.tensordot(Q, self.W_Q, axes=((2), (0)))
        K_l = tf.tensordot(K, self.W_K, axes=((2), (0)))
        V_l = tf.tensordot(V, self.W_V, axes=((2), (0)))

        # Split heads.
        Q_h = tf.transpose(tf.reshape(Q_l, (batch_size, -1, self.num_head, self.d_k_h)), perm=[0, 2, 1, 3])
        K_h = tf.transpose(tf.reshape(K_l, (batch_size, -1, self.num_head, self.d_k_h)), perm=[0, 2, 1, 3])
        V_h = tf.transpose(tf.reshape(V_l, (batch_size, -1, self.num_head, self.d_v_h)), perm=[0, 2, 1, 3])

        # Do self-attention according to similarity type.
        if self.similarity_type == SIMILARITY_TYPE_DIFF_ABS:
            head = tf.matmul(tf.nn.softmax(tf.math.exp(-1.0 * tf.abs(Q_h - tf.transpose(K_h, (0, 1, 3, 2))))) #* M)
                             , V_h)
        elif self.similarity_type == SIMILARITY_TYPE_PLAIN:
            head = tf.matmul(tf.nn.softmax(tf.matmul(Q_h, K_h, transpose_b=True)) #* M)
                             , V_h)
        elif self.similarity_type == SIMILARITY_TYPE_SCALED:
            head = tf.matmul(tf.nn.softmax(tf.matmul(Q_h, K_h, transpose_b=True) / np.sqrt(self.d_k)) #* M) #?
                             , V_h)
        elif self.similarity_type == SIMILARITY_TYPE_GENERAL:
            head = tf.matmul(tf.nn.softmax(tf.matmul(Q_h
                             , tf.tensordot(K_h, self.W_gen_S, axes=((3), (0))), transpose_b=True)) # * M)
                             , V_h)
        elif self.similarity_type == SIMILARITY_TYPE_ADDITIVE: #?
            head = tf.matmul(tf.nn.softmax((tf.tensordot(Q_h, self.W_add_S_Q, axes=((3), (0))) \
                   + tf.tensordot(K_h, self.W_add_S_K, axes=((3), (0)))) ) #* M)
                             , V_h)
        else:
            raise ValueError('similarity_type is not valid.')

        head = tf.transpose(head, perm=[0, 2, 1, 3])
        head = tf.reshape(head, (batch_size, -1, self.d_v))

        # Outputs.
        outputs = tf.tensordot(head, self.W_multi_head, axes=((2), (0)))

        return outputs

    def get_config(self):
        """Get configuration."""
        config = {'num_head': self.num_head
            , 'd_output': self.d_output
            , 'dropout_rate': self.dropout_rate
            , 'similarity_type': self.similarity_type}
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], input_shape[0][1], self.d_output)

        return output_shape