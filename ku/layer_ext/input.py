from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.python.keras.layers import Layer, InputSpec
import tensorflow.keras.initializers as initializers

from ku.backend_ext import tensorflow_backend as Ke

class InputVariable(Layer): #?
    """Input variable."""
    
    def __init__(self
                 , shape
                 , variable_initializer=initializers.Ones() #?
                 , **kwargs):
        self.shape=shape
        self.variable_initializer=initializers.get(variable_initializer)
        super(InputVariable, self).__init__(**kwargs)

    def build(self, input_shape):
        super(InputVariable, self).build(input_shape)
        self.variable_weight = self.add_weight(name='variable_weight'
                                 , shape=tuple(list(self.shape))
                                 , initializer=self.variable_initializer # Which initializer is optimal?
                                 , trainable=self.trainable) #?  

    def call(self, x):
        return self.variable_weight #?

    def get_config(self):
        config = {'shape': self.shape
            , 'variable_initializer':
                initializers.serialize(self.variable_initializer),
        }
        base_config = super(InputVariable, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0]] + list(self.shape))

class InputRandomUniform(Layer):
    """Uniformly random input."""
    
    def __init__(self
                 , shape
                 , minval=0.0
                 , maxval=0.0
                 , dtype=None
                 , seed=None
                 , **kwargs):
        self.shape=shape
        self.minval = minval
        self.maxval = maxval
        self.dtype = dtype
        self.seed = seed 
        super(InputRandomUniform, self).__init__(**kwargs)

    def build(self, input_shape):
        super(InputRandomUniform, self).build(input_shape)

    def call(self, x):        
        return K.random_uniform(shape=self.shape
                                , minval=self.minval
                                , maxval=self.maxval
                                , dtype=self.dtype
                                , seed=self.seed) #?

    def get_config(self):
        config = {'shape': self.shape
                  , 'minval': self.minval
                  , 'maxval': self.maxval
                  , 'dtype': self.dtype
                  , 'seed': self.seed
        }
        base_config = super(InputRandomUniform, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0]] + list(self.shape))
