from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.python.keras.layers import Layer, InputSpec
import tensorflow.keras.initializers as initializers

from ku.backend_ext import tensorflow_backend as Ke

class StyleMixingRegularization(_Merge): #?
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
                , lambda: K.random_uniform([], 1, num_layers, dtype=np.int32)
                , lambda: num_layers) #?
            d = Ke.where(Ke.broadcast_to(np.arange(num_layers)[np.newaxis, :, np.newaxis] \
                                        < cutoff, K.shape(d1)), d1, d2) #?
        else:
            d = d1
        
        return d

    def get_config(self):
        config = {'mixing_prob': self.mixing_prob}
        base_config = super(StyleMixingRegularization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
   
class TruncationTrick(Layer): #?
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
        super(TruncationTrick, self).build(input_shape)
        self.moving_mean = self.add_weight(shape=(input_shape[-1],) # Last channel?
                                           , name='moving_mean'
                                           , initializer=self.moving_mean_initializer
                                           , trainable=True) #?

    def call(self, x, training=None):
        def outputs_inference():
            # Apply truncation trick according to cutoff.
            num_layers = K.int_shape(x)[1]
            
            if self.cutoff is not None:
                beta = Ke.where(np.arange(num_layers)[np.newaxis, :, np.newaxis] < self.cutoff
                                , self.psi * np.ones(shape=(1, num_layers, 1), dtype=np.float32)
                                , np.ones(shape=(1, num_layers, 1), dtype=np.float32)) #?
            else:
                beta = np.ones(shape=(1, num_layers, 1), dtype=np.float32)
            
            return self.moving_mean + (x - self.moving_mean) * beta #?            
        
        # Update moving average.
        mean = K.mean(x[:, 0], axis=0) #?
        x_moving_mean = K.moving_average_update(self.moving_mean
                                , mean
                                , self.momentum) #? add_update?
        
        # Apply truncation trick according to cutoff.
        num_layers = K.int_shape(x)[1]
        
        if self.cutoff is not None:
            beta = Ke.where(np.arange(num_layers)[np.newaxis, :, np.newaxis] < self.cutoff
                            , self.psi * np.ones(shape=(1, num_layers, 1), dtype=np.float32)
                            , np.ones(shape=(1, num_layers, 1), dtype=np.float32)) #?
        else:
            beta = np.ones(shape=(1, num_layers, 1), dtype=np.float32)
    
        outputs = x_moving_mean + (x - self.moving_mean) * beta #?
        
        return K.in_train_phase(outputs, outputs_inference, training=training)

    def get_config(self):
        config = {'psi': self.psi
            , 'cutoff': self.cutoff
            , 'momentum': self.momentum
            , 'moving_mean_initializer':
                initializers.serialize(self.moving_mean_initializer)
        }
        base_config = super(TruncationTrick, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class MinibatchStddevConcat(Layer): #?
    """Minibatch standard deviation map concatenation layer."""

    def __init__(self
                 , group_size=4
                 , num_new_features=1
                 , **kwargs):
        super(MinibatchStddevConcat, self).__init__(**kwargs)
        self.group_size= group_size
        self.num_new_features = num_new_features

    def build(self, input_shape):
        super(MinibatchStddevConcat, self).build(input_shape)

    def call(self, x):
        group_size = tf.minimum(self.group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NHWC]  Input shape.
        y = tf.reshape(x
                       , [group_size
                          , -1
                          , s[1]
                          , s[2]
                          , s[3] // self.num_new_features
                          , self.num_new_features])
                          # [GMHWcn] Split minibatch into M groups of size G. Split channels into n channel groups c.
        y = tf.cast(y, tf.float32)                              # [GMHWcn] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMHWcn] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MHWcn]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MHWcn]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111n]  Take average over fmaps and pixels.
        y = tf.reduce_mean(y, axis=[3])                         # [M11n] Split channels into c channel groups
        y = tf.cast(y, x.dtype)                                 # [M11n]  Cast back to original data type.
        y = tf.tile(y, [group_size, s[1], s[2], 1])             # [NHWn]  Replicate over group and pixels.
        
        return tf.concat([x, y], axis=3)                        # [NHWC]  Append as new fmap.

    def get_config(self):
        config = {'group_size': self.group_size
            , 'num_new_features': self.num_new_features
        }
        base_config = super(MinibatchStddevConcat, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
