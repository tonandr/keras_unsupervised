"""
Created on 2019. 6. 21.

@author: Inwoo Chung (gutomitai@gmail.com)
License: BSD 3 clause.

Revision
    -Jun. 24, 2019
        AdaptiveIN is developed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.layers.merge import _Merge
from keras.layers import Layer, InputSpec

class AdaptiveIN(_Merge):
    """Adaptive instance normalization layer."""
    
    def __init__(self, axis=-1, epsilon=1e-6, **kwargs):
        # Check exception.
        if isinstance(axis, int) != True or axis == 0:
            raise ValueError('axis is a channel axis integer except for the batch axis.')
        
        self.axis = axis
        self.epsilon = epsilon 
        super(AdaptiveIN, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AdaptiveIN, self).build(input_shape)
        if len(input_shape) != 2:
            raise ValueError('A `AdaptiveIN` layer should be called '
                             'on exactly 2 inputs')
            
    def _merge_function(self, inputs):
        # Check exception.
        x = inputs
        if isinstance(x, list) != True or len(x) != 2:
            raise ValueError('Input must be a list of two tensors.')
        
        reduce_axis = tuple([i for i in range(1, K.ndim(x)) if i != self.axis])
        
        c = x[0] # Content tensor.
        s = x[1] # Style tensor.
        
        # Calculate mean and variance.
        c_mean = K.mean(c, axis=reduce_axis, keepdims=True)
        c_std = K.std(c, axis=reduce_axis, keepdims=True) + self.epsilon
        s_mean = K.mean(s, axis=reduce_axis, keepdims=True)
        s_std = K.std(s, axis=reduce_axis, keepdims=True)
        
        return s_std * ((c - c_mean) / c_std) + s_mean # Broadcasting?
                    
    def get_config(self):
        """Get configuration."""
        config = {'axis': self.axis
                  , 'epsilon': self.epsilon}
        base_config = super(AdaptiveIN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))    
