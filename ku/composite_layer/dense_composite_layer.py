from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Layer


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