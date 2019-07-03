"""
Created on 2019. 6. 27.

@author: Inwoo Chung (gutomitai@gmail.com)
License: BSD 3 clause.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from keras import backend as K
from keras.layers.merge import _Merge
from keras.layers import Layer, InputSpec
import keras.initializers as initializers

from ..backend_ext import tensorflow_backend as Ke

class StyleMixingRegularization(_Merge):
    """Style mixing regularization layer."""

    def __init__(self
                 , mixing_prob = None
                 , **kwargs):
        super(StyleMixingRegularization, self).__init__(**kwargs)
        self.mixing_prob = mixing_prob

    def build(self, input_shape):
        super(StyleMixingRegularization, self).build(input_shape)
        if len(input_shape) != 2:
            raise ValueError('A `StyleMixingRegularization` layer should be called '
                             'on exactly 2 inputs')

    def _merge_function(self, inputs):
        # Check exception.
        x = inputs
        if isinstance(x, list) != True or len(x) != 2:
            raise ValueError('Input must be a list of two tensors.')
                
        d1 = x[0] # Disentangled latent 1.
        d2 = x[1] # Disentangled latent 1.
        
        # Mixing style according to mixing probability.
        num_layers = K.int_shape(d1)[1]
        
        if self.mixing_prob is not None:
            cutoff = Ke.cond(
                K.random_uniform([], 0.0, 1.0) < self.mixing_prob
                , lambda: K.random_uniform([], 1, num_layers)
                , lambda: num_layers) #?
            d = Ke.where(Ke.broadcast_to(np.arange(num_layers)[np.newaxis, :, np.newaxis] \
                                        < cutoff, K.shape(d1)), d1, d2) #?
        else:
            d = d1
        
        return d

    def get_config(self):
        config = {'mixing_prob': self.mixing_prob}
        base_config = super(TruncationTrick, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
   
class TruncationTrick(Layer):
    """Truncation trick layer."""

    def __init__(self
                 , psi = 0.0
                 , cutoff = None
                 , momentum=0.99
                 , moving_mean_initializer='zeros'
                 , **kwargs):
        super(TruncationTrick, self).__init__(**kwargs)
        self.psi = psi
        self.cutoff = cutoff
        self.momentum = momentum
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)

    def build(self, input_shape):
        self.moving_mean = self.add_weight(shape=input_shape[-1] # Last channel?
                                           , name='moving_mean'
                                           , initializer=self.moving_mean_initializer
                                           , trainable=True) #?
        self.built = True

    def call(self, x):
        # Update moving average.
        mean = K.mean(x[:, 0], axis=0) #?
        K.moving_average_update(self.moving_mean
                                , mean
                                , self.momentum) #? add_update?
        
        # Apply truncation trick according to cutoff.
        num_layers = K.int_shape(x)[1]
        
        if self.cutoff is not None:
            beta = Ke.where(np.arange(num_layers)[np.newaxis, :, np.newaxis] < self.cutoff
                            , self.psi * np.ones(shape=(1, num_layers, 1))
                            , np.ones(shape=(1, num_layers, 1)))
        else:
            beta = np.ones(shape=(1, num_layers, 1))
    
        return self.moving_mean + (x - self.moving_mean) * beta

    def get_config(self):
        config = {'psi': self.psi
            , 'cutoff': self.cutoff
            , 'momentum': self.momentum
            , 'moving_mean_initializer':
                initializers.serialize(self.moving_mean_initializer),
        }
        base_config = super(TruncationTrick, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape