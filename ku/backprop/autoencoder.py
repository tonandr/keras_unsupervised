from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers

from tensorflow_core.python.keras.layers.convolutional import Conv

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
    input_r = output_layer.output
    
    # Reconstruct the model reversely.
    output = _get_reversed_outputs(output_layer, input_r)
    
    return Model(inputs=output_layer.output, outputs=output)

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
    output_layer = Dense(1)
    
    in_node = output_layer.inbound_nodes[0]
    out_layer = in_node.outbound_layer
    
    if isinstance(out_layer, Dense):
        output = Dense(out_layer.input_shape[1]
                       , activation=out_layer.activation
                       , use_bias=out_layer.use_bias)(input_r) #?
        
        # Get an upper layer.
        upper_layers = in_node.inbound_layers[0]
        
        if len(upper_layers) == 1:
            upper_layer = upper_layers[0]
            return _get_reversed_outputs(upper_layer, output)
        else:
            return output
    elif isinstance(out_layer, Conv): #?
        if Conv.rank == 1:
            output = Dense(out_layer.input_shape[1], activation=out_layer.activation)(input_r) #?
            
            # Get an upper layer.
            upper_layers = in_node.inbound_layers[0]
            
            if len(upper_layers) == 1:
                upper_layer = upper_layers[0]
                return _get_reversed_outputs(upper_layer, output)
            else:
                return output

def make_autoencoder(model, hps):
    """Make autoencoder.
    
    Parameters
    ----------
    model: Keras model
        Undirected graphical model.
    hps: dict
        Hyper-parameters.
    
    Returns
    -------
    Autoencoder model
        Keras model.
    """
    
    # Check exception.?
    # Get a reverse model.
    r_model = reverse_model(model)
    
    inputs = model.inputs
    latents = model(inputs)
    output = r_model(latents)
    autoencoder = Model(inputs=inputs, outputs=[output]) #?
    
    opt = optimizers.Adam(lr=hps['lr']
                            , beta_1=hps['beta_1']
                            , beta_2=hps['beta_2']
                            , decay=hps['decay'])
    
     
    autoencoder.compile(optimizer=opt, loss='mse') #?    
    
    return autoencoder