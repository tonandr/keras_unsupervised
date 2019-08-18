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
    
    def __init__(self, input_shape,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='random_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 gain=np.sqrt(2), 
                 lrmul=1, 
                 **kwargs):
        self.input_shape = input_shape
        self.gain = gain
        self.lrmul = lrmul
        
        he_std = self.gain / np.sqrt(np.prod(input_shape[1:], axis=-1)) #?
        init_std = 1.0 / self.lrmul
        runtime_coeff = he_std * self.lrmul
        kernel_initializer = initializers.random_normal(0, init_std) * runtime_coeff
        
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
        
    def get_config(self):
        config = {'input_shape': self.input_shape
                  , 'gain': self.gain
                  , 'lrmul': self.lrmul
        }
        base_config = super(EqualizedLRDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0]] + list(self.shape))