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
from keras.layers import Layer, InputSpec, Dense, Conv2D, Conv2DTranspose, DepthwiseConv2D
from keras.layers.convolutional import _Conv
from keras.utils import conv_utils
import keras.initializers as initializers

from ku.backend_ext import tensorflow_backend as Ke

class _EqualizedLRConv(_Conv):
    """Equalized learning rate abstract convolution layer."""
    
    def __init__(self, filters
                 , kernel_size
                 , rank=None
                 , strides=1
                 , padding='valid'
                 , data_format=None
                 , dilation_rate=1
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
                        
        super(_EqualizedLRConv, self).__init__(rank
                                               , filters
                                               , kernel_size
                                               , strides=strides
                                               , padding=padding
                                               , data_format=data_format
                                               , dilation_rate=dilation_rate
                                               , activation=activation
                                               , use_bias=use_bias
                                               , kernel_initializer=kernel_initializer
                                               , bias_initializer=bias_initializer
                                               , kernel_regularizer=kernel_regularizer
                                               , bias_regularizer=bias_regularizer
                                               , activity_regularizer=activity_regularizer
                                               , kernel_constraint=kernel_constraint
                                               , bias_constraint=bias_constraint 
                                               , **kwargs)

    def build(self, input_shape): # Bias?
        super(_EqualizedLRConv, self).build(input_shape)
        
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
        
        if self.rank == 1:
            outputs = K.conv1d(
                inputs,
                self.kernel,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])
        if self.rank == 2:
            outputs = K.conv2d(
                inputs,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.rank == 3:
            outputs = K.conv3d(
                inputs,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs
        
    def get_config(self):
        config = {'gain': self.gain
                  , 'lrmul': self.lrmul
        }
        base_config = super(_EqualizedLRConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class EqualizedLRConv1D(_EqualizedLRConv):
    """Equalized learning rate 1d convolution layer."""
    
    def __init__(self
                , filters
                , kernel_size
                , strides=1
                , padding='valid'
                , data_format=None
                , dilation_rate=1
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
        if padding == 'causal':
            if data_format != 'channels_last':
                raise ValueError('When padding is casual, data format must be channel_last.')        
        super(EqualizedLRConv1D, self).__init__(filters 
                                                , kernel_size
                                                , rank=1
                                                , strides=strides
                                                , padding=padding
                                                , data_format=data_format
                                                , dilation_rate=dilation_rate
                                                , activation=activation
                                                , use_bias=use_bias
                                                , kernel_initializer=kernel_initializer
                                                , bias_initializer=bias_initializer
                                                , kernel_regularizer=kernel_regularizer
                                                , bias_regularizer=bias_regularizer
                                                , activity_regularizer=activity_regularizer
                                                , kernel_constraint=kernel_constraint
                                                , bias_constraint=bias_constraint 
                                                , **kwargs)

    def build(self, input_shape):
        super(EqualizedLRConv1D, self).build(input_shape)
        
    def get_config(self):
        base_config = super(EqualizedLRConv1D, self).get_config()
        base_config.pop('rank') #?
        return base_config

class EqualizedLRConv2D(_EqualizedLRConv):
    """Equalized learning rate 2d convolution layer."""
    
    def __init__(self
                , filters
                , kernel_size
                , strides=(1, 1)
                , padding='valid'
                , data_format=None
                , dilation_rate=(1, 1)
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
        super(EqualizedLRConv2D, self).__init__(filters                              
                                                , kernel_size
                                                , rank=2
                                                , strides=strides
                                                , padding=padding
                                                , data_format=data_format
                                                , dilation_rate=dilation_rate
                                                , activation=activation
                                                , use_bias=use_bias
                                                , kernel_initializer=kernel_initializer
                                                , bias_initializer=bias_initializer
                                                , kernel_regularizer=kernel_regularizer
                                                , bias_regularizer=bias_regularizer
                                                , activity_regularizer=activity_regularizer
                                                , kernel_constraint=kernel_constraint
                                                , bias_constraint=bias_constraint 
                                                , **kwargs)

    def build(self, input_shape):
        super(EqualizedLRConv2D, self).build(input_shape)
        
    def get_config(self):
        base_config = super(EqualizedLRConv2D, self).get_config()
        base_config.pop('rank') #?
        return base_config

class EqualizedLRConv3D(_EqualizedLRConv):
    """Equalized learning rate 3d convolution layer."""
    
    def __init__(self
                , filters
                , kernel_size
                , strides=(1, 1, 1)
                , padding='valid'
                , data_format=None
                , dilation_rate=(1, 1, 1)
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
        super(EqualizedLRConv3D, self).__init__(filters
                , kernel_size
                , rank=3
                , strides=strides
                , padding=padding
                , data_format=data_format
                , dilation_rate=dilation_rate
                , activation=activation
                , use_bias=use_bias
                , kernel_initializer=kernel_initializer
                , bias_initializer=bias_initializer
                , kernel_regularizer=kernel_regularizer
                , bias_regularizer=bias_regularizer
                , activity_regularizer=activity_regularizer
                , kernel_constraint=kernel_constraint
                , bias_constraint=bias_constraint
                , **kwargs)

    def build(self, input_shape):
        super(EqualizedLRConv3D, self).build(input_shape)
        
    def get_config(self):
        base_config = super(EqualizedLRConv3D, self).get_config()
        base_config.pop('rank') #?
        return base_config

class _FusedConv(_Conv):
    """Fused abstraction convolution layer."""
    
    def __init__(self 
                , filters
                , kernel_size
                , rank=None 
                , strides=1
                , padding='valid'
                , data_format=None
                , dilation_rate=1
                , activation=None
                , use_bias=True
                , kernel_initializer='glorot_uniform'
                , bias_initializer='zeros'
                , kernel_regularizer=None
                , bias_regularizer=None
                , activity_regularizer=None
                , kernel_constraint=None
                , bias_constraint=None
                , **kwargs):        
        super(_FusedConv, self).__init__(rank
                , filters
                , kernel_size
                , strides=strides
                , padding=padding
                , data_format=data_format
                , dilation_rate=dilation_rate
                , activation=activation
                , use_bias=use_bias
                , kernel_initializer=kernel_initializer
                , bias_initializer=bias_initializer
                , kernel_regularizer=kernel_regularizer
                , bias_regularizer=bias_regularizer
                , activity_regularizer=activity_regularizer
                , kernel_constraint=kernel_constraint
                , bias_constraint=bias_constraint
                , **kwargs)

    def build(self, input_shape):
        super(_FusedConv, self).build(input_shape)
        
    def call(self, inputs):
        if self.rank == 1:
            kernel = Ke.pad(self.kernel
                            , [[1,1], [0,0], [0,0]])
            kernel = Ke.add_n([kernel[1:]
                               , kernel[:-1]])
            outputs = K.conv1d(inputs
                , self.kernel
                , strides=self.strides[0]
                , padding=self.padding
                , data_format=self.data_format
                , dilation_rate=self.dilation_rate[0])
        if self.rank == 2:
            kernel = Ke.pad(self.kernel
                            , [[1,1], [1,1], [0,0], [0,0]])
            kernel = Ke.add_n([kernel[1:, 1:]
                               , kernel[:-1, 1:]
                               , kernel[1:, :-1]
                               , kernel[:-1, :-1]])
            outputs = K.conv2d(inputs
                , kernel
                , strides=self.strides
                , padding=self.padding
                , data_format=self.data_format
                , dilation_rate=self.dilation_rate)
        if self.rank == 3:
            kernel = Ke.pad(self.kernel
                            , [[1,1], [1,1], [1,1], [0,0], [0,0]])
            kernel = Ke.add_n([kernel[1:, 1:, 1:]
                               , kernel[1:, 1:, :-1]
                               , kernel[1:, :-1, 1:]
                               , kernel[1:, :-1, :-1]
                               , kernel[:-1, 1:, 1:]
                               , kernel[:-1, 1:, :-1]
                               , kernel[:-1, :-1, 1:]
                               , kernel[:-1, :-1, :-1]])
            outputs = K.conv3d(inputs
                , self.kernel
                , strides=self.strides
                , padding=self.padding
                , data_format=self.data_format
                , dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(outputs
                , self.bias
                , data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs        

    def get_config(self):
        base_config = super(EqualizedLRConv3D, self).get_config()
        base_config.pop('rank') #?
        return base_config
    
class FusedConv1D(_FusedConv):
    """Fused 1d convolution layer."""
    
    def __init__(self 
                , filters
                , kernel_size
                , strides=1
                , padding='valid'
                , data_format=None
                , dilation_rate=1
                , activation=None
                , use_bias=True
                , kernel_initializer='glorot_uniform'
                , bias_initializer='zeros'
                , kernel_regularizer=None
                , bias_regularizer=None
                , activity_regularizer=None
                , kernel_constraint=None
                , bias_constraint=None
                , **kwargs):        
        super(FusedConv1D, self).__init__(filters
                , kernel_size
                , strides=strides
                , rank=1
                , padding=padding
                , data_format=data_format
                , dilation_rate=dilation_rate
                , activation=activation
                , use_bias=use_bias
                , kernel_initializer=kernel_initializer
                , bias_initializer=bias_initializer
                , kernel_regularizer=kernel_regularizer
                , bias_regularizer=bias_regularizer
                , activity_regularizer=activity_regularizer
                , kernel_constraint=kernel_constraint
                , bias_constraint=bias_constraint
                , **kwargs)

    def build(self, input_shape):
        super(FusedConv1D, self).build(input_shape)

class FusedConv2D(_FusedConv):
    """Fused 2d convolution layer."""
    
    def __init__(self 
                , filters
                , kernel_size
                , strides=(1, 1)
                , padding='valid'
                , data_format=None
                , dilation_rate=(1, 1)
                , activation=None
                , use_bias=True
                , kernel_initializer='glorot_uniform'
                , bias_initializer='zeros'
                , kernel_regularizer=None
                , bias_regularizer=None
                , activity_regularizer=None
                , kernel_constraint=None
                , bias_constraint=None
                , **kwargs):        
        super(FusedConv2D, self).__init__(filters
                , kernel_size
                , rank=2
                , strides=strides
                , padding=padding
                , data_format=data_format
                , dilation_rate=dilation_rate
                , activation=activation
                , use_bias=use_bias
                , kernel_initializer=kernel_initializer
                , bias_initializer=bias_initializer
                , kernel_regularizer=kernel_regularizer
                , bias_regularizer=bias_regularizer
                , activity_regularizer=activity_regularizer
                , kernel_constraint=kernel_constraint
                , bias_constraint=bias_constraint 
                , **kwargs)

    def build(self, input_shape):
        super(FusedConv2D, self).build(input_shape)

class FusedConv3D(_FusedConv):
    """Fused 3d convolution layer."""
    
    def __init__(self 
                , filters
                , kernel_size
                , strides=(1, 1, 1)
                , padding='valid'
                , data_format=None
                , dilation_rate=(1, 1, 1)
                , activation=None
                , use_bias=True
                , kernel_initializer='glorot_uniform'
                , bias_initializer='zeros'
                , kernel_regularizer=None
                , bias_regularizer=None
                , activity_regularizer=None
                , kernel_constraint=None
                , bias_constraint=None
                , **kwargs):        
        super(FusedConv3D, self).__init__(filters
                , kernel_size
                , rank=3
                , strides=strides
                , padding=padding
                , data_format=data_format
                , dilation_rate=dilation_rate
                , activation=activation
                , use_bias=use_bias
                , kernel_initializer=kernel_initializer
                , bias_initializer=bias_initializer
                , kernel_regularizer=kernel_regularizer
                , bias_regularizer=bias_regularizer
                , activity_regularizer=activity_regularizer
                , kernel_constraint=kernel_constraint
                , bias_constraint=bias_constraint
                , **kwargs)

    def build(self, input_shape):
        super(FusedConv3D, self).build(input_shape)

class FusedConv2DTranspose(Conv2DTranspose):
    """Fused 2d transposed convolution layer."""
    
    def __init__(self 
                , filters
                , kernel_size
                , strides=(1, 1)
                , padding='valid'
                , data_format=None
                , dilation_rate=(1, 1)
                , activation=None
                , use_bias=True
                , kernel_initializer='glorot_uniform'
                , bias_initializer='zeros'
                , kernel_regularizer=None
                , bias_regularizer=None
                , activity_regularizer=None
                , kernel_constraint=None
                , bias_constraint=None
                , **kwargs):        
        super(FusedConv2DTranspose, self).__init__(filters
                , kernel_size
                , strides=strides
                , padding=padding
                , data_format=data_format
                , dilation_rate=dilation_rate
                , activation=activation
                , use_bias=use_bias
                , kernel_initializer=kernel_initializer
                , bias_initializer=bias_initializer
                , kernel_regularizer=kernel_regularizer
                , bias_regularizer=bias_regularizer
                , activity_regularizer=activity_regularizer
                , kernel_constraint=kernel_constraint
                , bias_constraint=bias_constraint
                , **kwargs)

    def build(self, input_shape):
        super(FusedConv2DTranspose, self).build(input_shape)

    def call(self, inputs):
        input_shape = K.shape(inputs)
        batch_size = input_shape[0]
        if self.data_format == 'channels_first':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        height, width = input_shape[h_axis], input_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides
        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        out_height = conv_utils.deconv_length(height
                                              , stride_h, kernel_h
                                              , self.padding
                                              , out_pad_h
                                              , self.dilation_rate[0])
        out_width = conv_utils.deconv_length(width
                                            , stride_w, kernel_w
                                            , self.padding
                                            , out_pad_w
                                            , self.dilation_rate[1])
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)

        kernel = Ke.transpose(self.kernel,[0, 1, 3, 2]) #?
        kernel = Ke.pad(kernel
                            , [[1,1], [1,1], [0,0], [0,0]])
        kernel = Ke.add_n([kernel[1:, 1:]
                               , kernel[:-1, 1:]
                               , kernel[1:, :-1]
                               , kernel[:-1, :-1]])        
        outputs = K.conv2d_transpose(inputs
            , self.kernel
            , output_shape
            , self.strides
            , padding=self.padding
            , data_format=self.data_format
            , dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(outputs
                , self.bias
                , data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

class BlurDepthwiseConv2D(DepthwiseConv2D): #?
    """Blur 2d depthwise convolution layer."""
    
    def __init__(self
                , kernel_size=(1, 1)
                , blur_kernel=[1, 2, 1]
                , strides=(1, 1)
                , padding='valid'
                , depth_multiplier=1
                , data_format=None
                , dilation_rate=(1, 1)
                , activation=None
                , use_bias=True
                , depthwise_initializer='glorot_uniform'
                , bias_initializer='zeros'
                , depthwise_regularizer=None
                , bias_regularizer=None
                , activity_regularizer=None
                , depthwise_constraint=None
                , bias_constraint=None
                , **kwargs):
        self.blur_kernel = blur_kernel      
        super(BlurDepthwiseConv2D, self).__init__(kernel_size=kernel_size
                , strides=strides
                , padding=padding
                , depth_multiplier=depth_multiplier
                , data_format=data_format
                , dilation_rate=dilation_rate
                , activation=activation
                , use_bias=use_bias
                , depthwise_initializer=depthwise_initializer
                , bias_initializer=bias_initializer
                , depthwise_regularizer=depthwise_regularizer
                , bias_regularizer=bias_regularizer
                , activity_regularizer=activity_regularizer
                , depthwise_constraint=depthwise_constraint
                , bias_constraint=bias_constraint
                , **kwargs)

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv2D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        
        blur_filter = np.asarray(self.blur_kernel, dtype='float32')
        blur_filter = blur_filter[:, np.newaxis] * blur_filter[np.newaxis, :]
        blur_filter /= np.sum(blur_filter)
        blur_filter = blur_filter[::-1, ::-1] #?
        blur_filter = blur_filter[:, :, np.newaxis, np.newaxis]
        blur_filter = np.tile(blur_filter, [1, 1, input_dim, self.depth_multiplier])
        self.depthwise_kernel = K.constant(blur_filter) #?
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,)
                                        , initializer=self.bias_initializer
                                        , name='bias'
                                        , regularizer=self.bias_regularizer
                                        , constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def get_config(self):
        config = {'blur_kernel': self.blur_kernel
        }
        base_config = super(_EqualizedLRConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))