from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.keras import optimizers

def reverse_undirected_model(model):
    """Reverse the undirected model.
    
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
    
    # Get all layers and extract input layers and output layers.
    layers = model.layers
    input_layers = [layer for layer in layers if isinstance(layer, InputLayer) == True]
    output_layer_names = [t.name.split('/')[0] for t in model.outputs]
    output_layers = [layer for layer in layers if layer.name in output_layer_names] #?
    
    # Reconstruct the model reversely.
    outputs = model.outputs
    input1 = outputs[0]
    layer = output_layers[0] 
    output = _reverse_output(layer, input1)
    
    return Model(inputs=[input1], outputs=[output])

def _reverse_output(layer, tensor):
    """Reverse output layers recursively. ?
    
    Parameters
    ----------
    layer: Keras layer
        Layer instance.
    tensor: Karas tensor
        Tensor instance.
    """
    
    # Check exception.?
    if isinstance(layer, Dense):
        tensor = Dense(layer.input_shape[0], activation=layer.activation)(tensor)
        
        # Get inbound nodes.
        i_nodes = layer._inbound_nodes
        
        if len(i_nodes) != 0:
            output = _reverse_output(i_nodes[0], tensor)
        else:
            return tensor
    else:
        # TODO?
        pass
    
    return output

def make_ae_from_ugm(model, hps):
    """Make autoencoder from undirected graphical model.?
    
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
    r_model = reverse_undirected_model(model)
    
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