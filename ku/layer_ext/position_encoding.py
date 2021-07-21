from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer, InputSpec


class OrdinalPositionEncoding(Layer):
    """Position encoding with the normalized ordinal number."""

    def __init__(self, num_total_seq, **kwargs):
        super(OrdinalPositionEncoding, self).__init__(**kwargs)
        self.num_total_seq = num_total_seq

    def build(self, input_shape):
        super(OrdinalPositionEncoding, self).build(input_shape)

        # Create the normalized ordinal position according to num_seq and num_total_seq.
        pos = tf.range(1, self.num_total_seq + 1) / self.num_total_seq
        self.pos = tf.expand_dims(tf.expand_dims(pos, axis=-1), axis=0)

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs):
        # Check exception.
        x = inputs
        assert len(K.int_shape(x)) == 3

        num_seq = tf.shape(inputs)[1]

        # Encode positions.
        pe = x + self.pos[:, :num_seq, :] # Broadcating?

        return pe

    def get_config(self):
        """Get configuration."""
        config = {'num_total_seq': self.num_total_seq}
        base_config = super(OrdinalPositionEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class PeriodicPositionEncoding(Layer):
    """Periodic position encoding."""

    def __init__(self, max_seq, d_f, base_n, **kwargs):
        super(PeriodicPositionEncoding, self).__init__(**kwargs)
        self.max_seq = max_seq
        self.d_f = d_f
        self.base_n = base_n

    def build(self, input_shape):
        super(PeriodicPositionEncoding, self).build(input_shape)

        # Create the periodic position.
        pos = np.expand_dims(np.arange(self.max_seq), axis=-1)
        pos_f = np.expand_dims(np.arange(self.d_f), axis=0)
        pos = pos / np.power(self.base_n, (2 * (pos_f // 2) / np.float32(self.d_f)))
        pos[:, 0::2] = np.sin(pos[:, 0::2])
        pos[:, 1::2] = np.cos(pos[:, 1::2])

        self.pos = tf.cast(np.expand_dims(pos, axis=0), dtype=tf.float32)

    def call(self, inputs):
        # Check exception.
        x = inputs
        assert len(K.int_shape(x)) == 3

        num_seq = tf.shape(inputs)[1]

        # Encode positions.
        pe = x + self.pos[:, :num_seq, :] # Broadcating?

        return pe

    def get_config(self):
        """Get configuration."""
        config = {'max_seq': self.max_seq
                  , 'd_f': self.d_f
                  , 'base_n': self.base_n}
        base_config = super(PeriodicPositionEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape