from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.python.keras.layers import Layer, InputSpec, Dense, Conv2D, Conv3D
from tensorflow.python.keras.layers import Conv2DTranspose, DepthwiseConv2D
from tensorflow.python.keras.layers.convolutional import Conv, SeparableConv
from tensorflow.python.keras.utils import conv_utils
import tensorflow.keras.initializers as initializers

from ku.backend_ext import tensorflow_backend as Ke


class _EqualizedLRConv(Conv):
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
                 , kernel_initializer=None
                 , bias_initializer=None
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
        he_std = self.gain / np.sqrt(np.prod(input_shape[1:], axis=-1)) #?
        init_std = 1.0 / self.lrmul
        self.runtime_coeff = he_std * self.lrmul
        
        self.kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev=init_std)
        super(_EqualizedLRConv, self).build(input_shape)

    def call(self, inputs, training=None):
        scaled_kernel = self.kernel * self.runtime_coeff
        
        if self.rank == 1:
            outputs = K.conv1d(
                inputs,
                scaled_kernel,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])
        if self.rank == 2:
            outputs = K.conv2d(
                inputs,
                scaled_kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.rank == 3:
            outputs = K.conv3d(
                inputs,
                scaled_kernel,
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
            outputs = self.activation(outputs)
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
                , kernel_initializer=None
                , bias_initializer=None
                , kernel_regularizer=None
                , bias_regularizer=None
                , activity_regularizer=None
                , kernel_constraint=None
                , bias_constraint=None
                , gain=np.sqrt(2) 
                , lrmul=1 
                , **kwargs):      
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
                                                , gain=gain
                                                , lrmul=lrmul 
                                                , **kwargs)

    def build(self, input_shape):
        super(EqualizedLRConv1D, self).build(input_shape)


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
                , kernel_initializer=None
                , bias_initializer=None
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
                                                , gain=gain
                                                , lrmul=lrmul  
                                                , **kwargs)

    def build(self, input_shape):
        super(EqualizedLRConv2D, self).build(input_shape)


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
                , kernel_initializer=None
                , bias_initializer=None
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
                , gain=gain
                , lrmul=lrmul 
                , **kwargs)

    def build(self, input_shape):
        super(EqualizedLRConv3D, self).build(input_shape)


class _FusedEqualizedLRConv(Conv):
    """Fused, equalized learning rate abstraction convolution layer."""
    
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
                , kernel_initializer=None
                , bias_initializer=None
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
        super(_FusedEqualizedLRConv, self).__init__(rank
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
        he_std = self.gain / np.sqrt(np.prod(input_shape[1:], axis=-1)) #?
        init_std = 1.0 / self.lrmul
        self.runtime_coeff = he_std * self.lrmul
        
        self.kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev=init_std)
        super(_FusedEqualizedLRConv, self).build(input_shape)

    def call(self, inputs): #?
        scaled_kernel = self.kernel * self.runtime_coeff
        
        if self.rank == 1:
            kernel = Ke.pad(scaled_kernel
                            , [[1,1], [0,0], [0,0]])
            fused_kernel = Ke.add_n([kernel[1:]
                               , kernel[:-1]]) / 2.0
            outputs = K.conv1d(inputs
                , fused_kernel
                , strides=self.strides[0]
                , padding=self.padding
                , data_format=self.data_format
                , dilation_rate=self.dilation_rate[0])
        if self.rank == 2:
            kernel = Ke.pad(scaled_kernel
                            , [[1,1], [1,1], [0,0], [0,0]])
            fused_kernel = Ke.add_n([kernel[1:, 1:]
                               , kernel[:-1, 1:]
                               , kernel[1:, :-1]
                               , kernel[:-1, :-1]]) / 4.0
            outputs = K.conv2d(inputs
                , fused_kernel
                , strides=self.strides
                , padding=self.padding
                , data_format=self.data_format
                , dilation_rate=self.dilation_rate)
        if self.rank == 3:
            kernel = Ke.pad(scaled_kernel
                            , [[1,1], [1,1], [1,1], [0,0], [0,0]])
            fused_kernel = Ke.add_n([kernel[1:, 1:, 1:]
                               , kernel[1:, 1:, :-1]
                               , kernel[1:, :-1, 1:]
                               , kernel[1:, :-1, :-1]
                               , kernel[:-1, 1:, 1:]
                               , kernel[:-1, 1:, :-1]
                               , kernel[:-1, :-1, 1:]
                               , kernel[:-1, :-1, :-1]]) / 8.0
            outputs = K.conv3d(inputs
                , fused_kernel
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
        config = {'gain': self.gain
                  , 'lrmul': self.lrmul
        }
        base_config = super(_EqualizedLRConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FusedEqualizedLRConv1D(_FusedEqualizedLRConv):
    """Fused, equalized learning rate 1d convolution layer."""
    
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
                , gain=np.sqrt(2) 
                , lrmul=1 
                , **kwargs):        
        super(FusedEqualizedLRConv1D, self).__init__(filters
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
                , gain=gain
                , lrmul=lrmul 
                , **kwargs)

    def build(self, input_shape):
        super(FusedEqualizedLRConv1D, self).build(input_shape)


class FusedEqualizedLRConv2D(_FusedEqualizedLRConv):
    """Fused, equalized learning rate 2d convolution layer."""
    
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
                , gain=np.sqrt(2) 
                , lrmul=1 
                , **kwargs):        
        super(FusedEqualizedLRConv2D, self).__init__(filters
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
                , gain=gain
                , lrmul=lrmul  
                , **kwargs)

    def build(self, input_shape):
        super(FusedEqualizedLRConv2D, self).build(input_shape)


class FusedEqualizedLRConv3D(_FusedEqualizedLRConv):
    """Fused, equalized learning rate 3d convolution layer."""
    
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
                , gain=np.sqrt(2) 
                , lrmul=1 
                , **kwargs):        
        super(FusedEqualizedLRConv3D, self).__init__(filters
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
                , gain=gain
                , lrmul=lrmul 
                , **kwargs)

    def build(self, input_shape):
        super(FusedEqualizedLRConv3D, self).build(input_shape)


class FusedEqualizedLRConv2DTranspose(Conv2DTranspose): #?
    """Fused, equalized learning rate 2d transposed convolution layer."""
    
    def __init__(self 
                , filters
                , kernel_size
                , strides=(1, 1)
                , padding='valid'
                , data_format=None
                , dilation_rate=(1, 1)
                , activation=None
                , use_bias=True
                , kernel_initializer=None
                , bias_initializer=None
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
        super(FusedEqualizedLRConv2DTranspose, self).__init__(filters
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
        he_std = self.gain / np.sqrt(np.prod(input_shape[1:], axis=-1)) #?
        init_std = 1.0 / self.lrmul
        self.runtime_coeff = he_std * self.lrmul
        
        self.kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev=init_std)
        super(FusedEqualizedLRConv2DTranspose, self).build(input_shape)

    def call(self, inputs):
        input_shape = K.shape(inputs)
        batch_size = input_shape[0]
        if self.data_format == 'channels_first': #?
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
        out_height = conv_utils.deconv_output_length(height
                                              , kernel_h
                                              , self.padding
                                              , output_padding=out_pad_h
                                              , stride=stride_h
                                              , dilation=self.dilation_rate[0])
        out_width = conv_utils.deconv_output_length(width
                                            , kernel_w
                                            , self.padding
                                            , output_padding=out_pad_w
                                            , stride=stride_w
                                            , dilation=self.dilation_rate[1])
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)

        scaled_kernel = self.kernel * self.runtime_coeff
        kernel = Ke.transpose(scaled_kernel,[0, 1, 3, 2]) #?
        kernel = Ke.pad(kernel
                            , [[1,1], [1,1], [0,0], [0,0]]) 
        fused_kernel = Ke.add_n([kernel[1:, 1:]
                               , kernel[:-1, 1:]
                               , kernel[1:, :-1]
                               , kernel[:-1, :-1]]) #?       
        outputs = K.conv2d_transpose(inputs
            , fused_kernel
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
                , blur_kernel=[1, 2, 1]
                , kernel_size=(3, 3)
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
        self.kernel_size = blur_filter.shape[:2]
        
        def depthwise_init(shape, dtype=None):
            return K.constant(blur_filter)
        
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=depthwise_init,
            name='blur_depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint,
            trainable=True)
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,)
                                        , initializer=self.bias_initializer
                                        , name='bias'
                                        , regularizer=self.bias_regularizer
                                        , constraint=self.bias_constraint
                                        , trainable=False)
        else:
            self.bias = None
            
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def get_config(self):
        config = {'blur_kernel': self.blur_kernel
        }
        base_config = super(BlurDepthwiseConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DepthwiseConv3D(DepthwiseConv2D):  # ?
    """3d depthwise convolution layer."""

    def __init__(self
                 , kernel_size
                 , strides=(1, 1, 1)
                 , padding='valid'
                 , depth_multiplier=1
                 , data_format=None
                 , dilation_rate=(1, 1, 1)
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
        super(DepthwiseConv3D, self).__init__(kernel_size=kernel_size
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
        if len(input_shape) < 5:
            raise ValueError('Inputs to `DepthwiseConv3D` should have rank 5. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 4
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv3D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        self.depthwise_kernels = []

        for c in range(input_dim * self.depth_multiplier):
            depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  self.kernel_size[2])

            self.depthwise_kernels.append(self.add_weight(
                shape=depthwise_kernel_shape,
                initializer=self.depthwise_initializer,
                name='depthwise_kernel_' + str(c),
                regularizer=self.depthwise_regularizer,
                constraint=self.depthwise_constraint,
                trainable=True))

        if self.use_bias:
            self.biases = []

            for c in range(input_dim * self.depth_multiplier):
                self.biases.append(self.add_weight(shape=(1,)
                                        , initializer=self.bias_initializer
                                        , name='bias_' + str(c)
                                        , regularizer=self.bias_regularizer
                                        , constraint=self.bias_constraint
                                        , trainable=False))
        else:
            self.biases = None

        # Set input spec.
        self.input_spec = InputSpec(ndim=5, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        outputs = []

        if self.data_format == 'channels_first':
            count = 0

            for c in range(self.input_spec.axes[1]):
                input = inputs[:, c:c+1, ...]

                for d in range(self.depth_multiplier):
                    output = K.conv3d(input
                                      , self.depthwise_kernels[count]
                                      , padding=self.padding
                                      , data_format=self.data_format
                                      , dilation_rate=self.dilation_rate)

                    if self.use_bias:
                        output = K.bias_add(output
                                            , self.biases[count]
                                            , data_format=self.data_format)

                    outputs.append(output)
                    count +=1

            outputs = K.concatenate(outputs, axis=1)
        else:
            count = 0

            for c in range(self.input_spec.axes[4]):
                input = inputs[:, c:c + 1, ...]

                for d in range(self.depth_multiplier):
                    output = K.conv3d(input
                                      , self.depthwise_kernels[count]
                                      , padding=self.padding
                                      , data_format=self.data_format
                                      , dilation_rate=self.dilation_rate)

                    if self.use_bias:
                        output = K.bias_add(output
                                            , self.biases[count]
                                            , data_format=self.data_format)

                    outputs.append(output)
                    count += 1

            outputs = K.concatenate(outputs, axis=4)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            hgts = input_shape[4]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            hgts = input_shape[3]
            out_filters = input_shape[4] * self.depth_multiplier

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0],
                                             self.dilation_rate[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1],
                                             self.dilation_rate[1])
        hgts = conv_utils.conv_output_length(cols, self.kernel_size[2],
                                             self.padding,
                                             self.strides[2],
                                             self.dilation_rate[2])

        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, rows, cols, hgts)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, hgts, out_filters)

    def get_config(self):
        config = {}
        base_config = super(DepthwiseConv3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SeparableConv3D(SeparableConv):
    """3d separable convolution layer."""

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 depth_multiplier=1,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 pointwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 pointwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 pointwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(SeparableConv3D, self).__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            depth_multiplier=depth_multiplier,
            activation=activations.get(activation),
            use_bias=use_bias,
            depthwise_initializer=initializers.get(depthwise_initializer),
            pointwise_initializer=initializers.get(pointwise_initializer),
            bias_initializer=initializers.get(bias_initializer),
            depthwise_regularizer=regularizers.get(depthwise_regularizer),
            pointwise_regularizer=regularizers.get(pointwise_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            depthwise_constraint=constraints.get(depthwise_constraint),
            pointwise_constraint=constraints.get(pointwise_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)

    def build(self, input_shape):
        if len(input_shape) < 5:
            raise ValueError('Inputs to `SeparableConv3D` should have rank 5. '
                             'Received input shape:', str(input_shape))
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`SeparableConv3D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        self.depthwise_kernels = []

        for c in range(input_dim * self.depth_multiplier):
            depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  self.kernel_size[2])

            self.depthwise_kernels.append(self.add_weight(
                shape=depthwise_kernel_shape,
                initializer=self.depthwise_initializer,
                name='depthwise_kernel_' + str(c),
                regularizer=self.depthwise_regularizer,
                constraint=self.depthwise_constraint,
                trainable=True,
                dtype=self.dtype))

        pointwise_kernel_shape = (1,) * self.rank + (self.depth_multiplier * input_dim, self.filters)
        self.pointwise_kernel = self.add_weight(
            name='pointwise_kernel',
            shape=pointwise_kernel_shape,
            initializer=self.pointwise_initializer,
            regularizer=self.pointwise_regularizer,
            constraint=self.pointwise_constraint,
            trainable=True,
            dtype=self.dtype)

        if self.use_bias:
            self.biases = []

            for c in range(input_dim * self.depth_multiplier):
                self.biases.append(self.add_weight(shape=(1,)
                                        , initializer=self.bias_initializer
                                        , name='bias_' + str(c)
                                        , regularizer=self.bias_regularizer
                                        , constraint=self.bias_constraint
                                        , trainable=False))
        else:
            self.biases = None

        # Set input spec.
        self.input_spec = InputSpec(ndim=5, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        outputs = []

        if self.data_format == 'channels_first':
            count = 0

            for c in range(self.input_spec.axes[1]):
                input = inputs[:, c:c+1, ...]

                for d in range(self.depth_multiplier):
                    output = K.conv3d(input
                                      , self.depthwise_kernels[count]
                                      , padding=self.padding
                                      , data_format=self.data_format
                                      , dilation_rate=self.dilation_rate)

                    if self.use_bias:
                        output = K.bias_add(output
                                            , self.biases[count]
                                            , data_format=self.data_format)

                    outputs.append(output)
                    count +=1

            outputs = K.concatenate(outputs, axis=1)
        else:
            count = 0

            for c in range(self.input_spec.axes[4]):
                input = inputs[:, c:c + 1, ...]

                for d in range(self.depth_multiplier):
                    output = K.conv3d(input
                                      , self.depthwise_kernels[count]
                                      , padding=self.padding
                                      , data_format=self.data_format
                                      , dilation_rate=self.dilation_rate)

                    if self.use_bias:
                        output = K.bias_add(output
                                            , self.biases[count]
                                            , data_format=self.data_format)

                    outputs.append(output)
                    count += 1

            outputs = K.concatenate(outputs, axis=4)

        outputs = K.conv3d(outputs
                           , self.pointwise_kernel
                           , padding=self.padding
                           , data_format=self.data_format
                           , dilation_rate=self.dilation_rate)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = {}
        base_config = super(SeparableConv3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
