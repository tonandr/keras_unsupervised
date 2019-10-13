from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.python.keras.layers import Layer, InputSpec, Dense
import tensorflow.keras.initializers as initializers

from ku.backend_ext import tensorflow_backend as Ke

class EqualizedLRDense(Dense):
    """Equalized learning rate dense layer."""
    
    def __init__(self
                , units
                , activation=None
                , use_bias=True
                , kernel_initializer='he_normal'
                , bias_initializer='he_normal'
                , kernel_regularizer=None
                , bias_regularizer=None
                , activity_regularizer=None
                , kernel_constraint=None
                , bias_constraint=None
                , gain=np.sqrt(2)
                , lrmul=1
                , **kwargs):
        self.gain = gain
        self.lrmul = lrmul
        
        super(EqualizedLRDense, self).__init__(units,
                 activation=activation,
                 use_bias=use_bias,
                 kernel_initializer=kernel_initializer,
                 bias_initializer=bias_initializer,
                 kernel_regularizer=kernel_regularizer,
                 bias_regularizer=bias_regularizer,
                 activity_regularizer=activity_regularizer,
                 kernel_constraint=kernel_constraint,
                 bias_constraint=bias_constraint, 
                 **kwargs)

    def build(self, input_shape):
        he_std = self.gain / np.sqrt(np.prod(input_shape[1:], axis=-1)) #?
        init_std = 1.0 / self.lrmul
        self.runtime_coeff = he_std * self.lrmul
        
        def he_init(shape, dtype=None):
            return K.random_normal(shape, mean=0., stddev=init_std)
        
        self.kernel_initializer = he_init
        super(EqualizedLRDense, self).build(input_shape)

    def call(self, inputs):                    
        scaled_kernel = self.kernel * self.runtime_coeff
        outputs = K.dot(inputs, scaled_kernel)

        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format='channels_last')
        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs
                
    def get_config(self):
        config = {'gain': self.gain
                  , 'lrmul': self.lrmul
        }
        base_config = super(EqualizedLRDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))