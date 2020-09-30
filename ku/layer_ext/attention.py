from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.python.keras.layers import Layer, InputSpec, Dense, Conv2D, Conv3D
from tensorflow.python.keras.layers import Conv2DTranspose, DepthwiseConv2D
from tensorflow.python.keras.layers.convolutional import Conv, SeparableConv
from tensorflow.python.keras.utils import conv_utils
import tensorflow.keras.initializers as initializers

# Constants.
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
                        or (similarity_type in [SIMILARITY_TYPE_PLAIN
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
        super(MultiHeadAttention, self).build(input_shape)
        if len(input_shape) != 4:
            raise ValueError('MultiHeadAttention layer should be called '
                             'on four Q, K, V, M inputs')

        # Get dimension of K, V.
        self.d_seq = input_shape[1][-2]
        self.d_k = input_shape[1][-1]
        self.d_v = input_shape[2][-1]

        # Create linear weight tensors of Q, K, V for each head.
        self.W_Qs = [self.add_weight(name='Q_linear_weight_' + str(i)
                                 , shape=(self.d_output, self.d_k)
                                 , initializer='truncated_normal' # Which initializer is optimal?
                                 , trainable=True) for i in range(self.num_head)]
        self.W_Ks = [self.add_weight(name='K_linear_weight_' + str(i)
                                 , shape=(self.d_output, self.d_k)
                                 , initializer='truncated_normal' # Which initializer is optimal?
                                 , trainable=True) for i in range(self.num_head)]
        self.W_Vs = [self.add_weight(name='V_linear_weight_' + str(i)
                                 , shape=(self.d_output, self.d_v)
                                 , initializer='truncated_normal' # Which initializer is optimal?
                                 , trainable=True) for i in range(self.num_head)]

        # Create weight tensors for similarity calculation.
        if self.similarity_type == SIMILARITY_TYPE_GENERAL:
            self.W_gen_Ss = [self.add_weight(name='general_similarity_weight_' + str(i)
                                 , shape=(self.d_k, self.d_k)
                                 , initializer='truncated_normal' # Which initializer is optimal?
                                 , trainable=True) for i in range(self.num_head)]
        elif self.similarity_type == SIMILARITY_TYPE_ADDITIVE:
            self.W_add_S_Qs = [self.add_weight(name='Q_additive_similarity_weight_' + str(i)
                                 , shape=(self.d_k, self.d_seq)
                                 , initializer='truncated_normal' # Which initializer is optimal?
                                 , trainable=True) for i in range(self.num_head)]
            self.W_add_S_Ks = [self.add_weight(name='K_additive_similarity_weight_' + str(i)
                                 , shape=(self.d_k, self.d_seq)
                                 , initializer='truncated_normal' # Which initializer is optimal?
                                 , trainable=True) for i in range(self.num_head)]

        # Create the multi-head weight.
        self.W_multi_head = self.add_weight(name='multi_head_weight'
                                 , shape=(int(self.num_head * self.d_v), self.d_output)
                                 , initializer='truncated_normal' # Which initializer is optimal?
                                 , trainable=True)

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

        # Do self-attention according to the number of heads and a similarity type.
        heads = []

        for i in range(self.num_head):
            # Do linear transformation.
            Q_h = tf.tensordot(Q, self.W_Qs[i], axes=((2), (0)))
            K_h = tf.tensordot(K, self.W_Ks[i], axes=((2), (0)))
            V_h = tf.tensordot(V, self.W_Vs[i], axes=((2), (0)))

            # Do self-attention according to similarity type.
            if self.similarity_type == SIMILARITY_TYPE_PLAIN:
                head = tf.matmul(tf.nn.softmax(tf.matmul(Q_h, K_h, transpose_b=True) * M)
                                 , V_h)
            elif self.similarity_type == SIMILARITY_TYPE_SCALED:
                head = tf.matmul(tf.nn.softmax(tf.matmul(Q_h, K_h, transpose_b=True) / np.sqrt(self.d_k) * M) #?
                                 , V_h)
            elif self.similarity_type == SIMILARITY_TYPE_GENERAL:
                head = tf.matmul(tf.nn_softmax(tf.matmul(Q_h
                                 , tf.tensordot(K_h, self.W_gen_Ss[i], axes=((2), (0))), transpose_b=True) * M)
                                 , V_h)
            elif self.similarity_type == SIMILARITY_TYPE_ADDITIVE:
                head = tf.matmul(tf.nn_softmax((tf.tensordot(Q_h, self.W_add_S_Qs[i], axes=((2), (0))) \
                       + tf.tensordot(K_h, self.W_add_S_Ks[i], axes=((2), (0)))) * M)
                                 , V_h)
            else:
                raise ValueError('similarity_type is not valid.')

            heads.append(head)

        # Concatenate heads.
        heads = tf.concat(heads, axis=-1)

        # Outputs.
        if training:
            outputs = tf.nn.dropout(tf.tensordot(heads, self.W_multi_head, axes=((2), (0))), rate=self.dropout_rate)
        else:
            outputs = tf.tensordot(heads, self.W_multi_head, axes=((2), (0)))

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