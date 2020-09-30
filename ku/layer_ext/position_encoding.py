from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer, InputSpec


class OrdinalPositionEncoding(Layer):
    """Position encoding with the normalized ordinal number."""

    def __init__(self, num_seq, num_total_seq, **kwargs):
        super(OrdinalPositionEncoding, self).__init__(**kwargs)
        self.num_seq = num_seq
        self.num_total_seq = num_total_seq

    def build(self, input_shape):
        super(OrdinalPositionEncoding, self).build(input_shape)

        # Create the normalized ordinal position according to num_seq and num_total_seq.
        pos = tf.range(1, self.num_seq + 1) / self.num_total_seq
        self.pos = tf.expand_dims(tf.expand_dims(pos, axis=-1), axis=0)

    def call(self, inputs):
        # Check exception.
        x = inputs
        assert len(K.int_shape(x)) == 3

        # Encode positions.
        pe = x + self.pos # Broadcating?

        return pe

    def get_config(self):
        """Get configuration."""
        config = {'num_total_seq': self.num_total_seq}
        base_config = super(OrdinalPositionEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape