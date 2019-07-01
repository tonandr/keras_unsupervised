"""
Created on 2019. 6. 12.

@author: Inwoo Chung (gutomitai@gmail.com)
License: BSD 3 clause.

Revision:
"""

from keras.models import Model
from keras.layers import Input, Dense
from keras.engine.input_layer import InputLayer

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
    layers = model.layers()
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