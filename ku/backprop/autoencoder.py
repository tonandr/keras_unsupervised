from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, Conv3D\
    , SeparableConv1D, SeparableConv2D, Conv2DTranspose, Conv3DTranspose\
    , Activation, BatchNormalization, InputLayer, Flatten, LeakyReLU
from tensorflow.keras import optimizers

from ..engine_ext import ModelExt

def reverse_model(model):
    """Reverse a model.
    
    Shared layer, multiple nodes ?
    
    Parameters
    ----------
    model: Keras model
        Model instance.
    
    Returns
    -------
    Keras model
        Reversed model instance.
    """
    
    # Check exception.
    # TODO
    
    # Get all layers and extract the input layer and output layer.
    layers = model.layers
    output_layer = layers[-1]
    input_r = tf.keras.Input(shape=K.int_shape(output_layer.output)[1:])  
    
    # Reconstruct the model reversely.
    output = _get_reversed_outputs(output_layer, input_r)
    
    return Model(inputs=input_r, outputs=output)

def _get_reversed_outputs(output_layer, input_r):
    """Get reverse outputs recursively. ?
    
    Parameters
    ----------
    output_layer: Keras layer.
        Last layer of a model.
    input_r: Tensor.
        Reversed input.
    """
    
    # Check exception.?
    # TODO
    
    in_node = output_layer.inbound_nodes[0]
    out_layer = in_node.outbound_layer
    
    if isinstance(out_layer, InputLayer):
        output = input_r
        return output    
    elif isinstance(out_layer, Dense):
        output = Dense(out_layer.input_shape[1]
                       , activation=out_layer.activation
                       , use_bias=out_layer.use_bias)(input_r) #?
        
        # Get an upper layer.
        upper_layer = in_node.inbound_layers
        return _get_reversed_outputs(upper_layer, output)
    elif isinstance(out_layer, (Conv1D, SeparableConv1D)): #?
        # TODO
        pass
    elif isinstance(out_layer, (Conv2D, SeparableConv2D)):
        output = Conv2DTranspose(out_layer.input_shape[-1]
                                     , out_layer.kernel_size
                                     , strides=out_layer.strides
                                     , padding='same' #?
                                     , activation=out_layer.activation
                                     , use_bias=out_layer.use_bias)(input_r) #?
        # Get an upper layer.
        upper_layer = in_node.inbound_layers
        return _get_reversed_outputs(upper_layer, output)
    elif isinstance(out_layer, (Conv3D)):
        output = Conv3DTranspose(out_layer.input_shape[-1]
                                     , out_layer.kernel_size
                                     , strides=out_layer.strides
                                     , padding='same' #?
                                     , activation=out_layer.activation
                                     , use_bias=out_layer.use_bias)(input_r) #?
            
        # Get an upper layer.
        upper_layer = in_node.inbound_layers
        return _get_reversed_outputs(upper_layer, output)
    elif isinstance(out_layer, Activation):
        output = Activation(out_layer.activation)(input_r) #?
            
        # Get an upper layer.
        upper_layer = in_node.inbound_layers
        return _get_reversed_outputs(upper_layer, output)
    elif isinstance(out_layer, LeakyReLU):
        output = LeakyReLU(out_layer.alpha)(input_r) #?
            
        # Get an upper layer.
        upper_layer = in_node.inbound_layers
        return _get_reversed_outputs(upper_layer, output)    
    elif isinstance(out_layer, BatchNormalization):
        output = BatchNormalization(axis=out_layer.axis
                                        , momentum=out_layer.momentum
                                        , epsilon=out_layer.epsilon)(input_r) #?
            
        # Get an upper layer.
        upper_layer = in_node.inbound_layers
        return _get_reversed_outputs(upper_layer, output) 
    else:
        raise RuntimeError('Layers must be supported in layer reversing.')
                    
def make_autoencoder_with_encoder(encoder, name=None):
    """Make autoencoder with encoder.
    
    Parameters
    ----------
    encoder: Keras model
        Encoder.
    name: String.
        Autoencoder model's name.
    
    Returns
    -------
    Autoencoder model
        Keras model.
    """
    
    # Check exception.?
    # Get a reverse model.
    r_model = reverse_model(encoder)
    
    inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in encoder.inputs]  
    latents = encoder(inputs)
    output = r_model(latents)    
    return Model(inputs=inputs, outputs=[output], name=name) #?