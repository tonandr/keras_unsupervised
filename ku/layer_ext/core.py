"""
Created on 2019. 8. 16.

@author: Inwoo Chung (gutomitai@gmail.com)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from keras import backend as K
from keras.layers.merge import _Merge
from keras.layers import Layer, InputSpec, Dense
import keras.initializers as initializers

from ku.backend_ext import tensorflow_backend as Ke

class EqualizedLRDense(Dense):
    """Equalized learning rate dense layer."""
    
    def __init__(self
                , units
                , activation=None
                , use_bias=True
                , kernel_initializer='random_normal'
                , bias_initializer='random_normal'
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
        super(EqualizedLRDense, self).build(input_shape)

        self.he_std = self.gain / np.sqrt(np.prod(input_shape[1:], axis=-1)) #?
        self.init_std = 1.0 / self.lrmul
        self.runtime_coeff = self.he_std * self.lrmul
        
        he_const = K.constant(np.random.normal(0
                                               , self.init_std * self.runtime_coeff
                                               , size=K.int_shape(self.kernel)))
        self.elr_normalized_kernel_func = K.function([self.kernel, he_const], [self.kernel / he_const])

    def call(self, inputs):
        he_const = np.random.normal(0, self.init_std * self.runtime_coeff, size=K.int_shape(self.kernel))
        elr_normalized_kernel = self.elr_normalized_kernel_func([K.get_session().run(self.kernel.value()), he_const])
        self.kernel.assign(elr_normalized_kernel[0])
        
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output
        
    def get_config(self):
        config = {'gain': self.gain
                  , 'lrmul': self.lrmul
        }
        base_config = super(EqualizedLRDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))