from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.python.keras.layers.merge import _Merge

class AdaptiveINWithStyle(Layer):
    """Adaptive instance normalization layer with the image and disentangled latent tensors."""
    
    def __init__(self, axis=-1, epsilon=1e-7, **kwargs):
        # Check exception.
        if isinstance(axis, int) != True or axis == 0 :
            raise ValueError('axis is a channel axis integer except for the batch axis.')
        
        self.axis = axis
        self.epsilon = epsilon
        super(AdaptiveINWithStyle, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AdaptiveINWithStyle, self).build(input_shape)
        if len(input_shape) != 2:
            raise ValueError('A `AdaptiveINWithStyle` layer should be called '
                             'on exactly 2 inputs')
            
    def call(self, inputs):
        # Check exception.
        x = inputs
        if isinstance(x, list) != True or len(x) != 2:
            raise ValueError('Input must be a list of two tensors.')
        assert len(K.int_shape(x[1])) == 2
        
        if self.axis < 0:
            assert K.ndim(x[0]) + self.axis >= 0
            self.axis = K.ndim(x[0]) + self.axis #?
            
        reduce_axes = tuple([i for i in range(1, K.ndim(x[0])) if i != self.axis])
        
        c = x[0] # Content image tensor.
        s = x[1] # Style dlatent tensor.
        
        # Calculate mean and variance.
        c_mean = K.mean(c, axis=reduce_axes, keepdims=True)
        c_std = K.std(c, axis=reduce_axes, keepdims=True) + self.epsilon
        s = K.reshape(s, [-1, 2, 1, 1, c.shape[-1]]) #?
        
        return (s[:, 0] + 1) * ((c - c_mean) / c_std) + s[:, 1]
                    
    def get_config(self):
        """Get configuration."""
        config = {'axis': self.axis
                   , 'epsilon': self.epsilon}
        base_config = super(AdaptiveINWithStyle, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))  

    def compute_output_shape(self, input_shape):
        return input_shape

class AdaptiveIN(_Merge):
    """Adaptive instance normalization layer."""
    
    def __init__(self, axis=-1, epsilon=1e-7, **kwargs):
        # Check exception.
        if isinstance(axis, int) != True or axis == 0 :
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
        assert K.int_shape(x[0]) == K.int_shape(x[1]) #?
        
        if self.axis < 0:
            assert K.ndim(x[0]) + self.axis >= 0
            self.axis = K.ndim(x[0]) + self.axis #?
            
        reduce_axes = tuple([i for i in range(1, K.ndim(x[0])) if i != self.axis])
        
        c = x[0] # Content image tensor.
        s = x[1] # Style image tensor.
        
        # Calculate mean and variance.
        c_mean = K.mean(c, axis=reduce_axes, keepdims=True)
        c_std = K.std(c, axis=reduce_axes, keepdims=True) + self.epsilon
        s_mean = K.mean(s, axis=reduce_axes, keepdims=True)
        s_std = K.std(s, axis=reduce_axes, keepdims=True)
        
        return s_std * ((c - c_mean) / c_std) + s_mean # Broadcasting?
                    
    def get_config(self):
        """Get configuration."""
        config = {'axis': self.axis
                   , 'epsilon': self.epsilon}
        base_config = super(AdaptiveIN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))    