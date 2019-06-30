"""
Created on 2019. 6. 12.

@author: Inwoo Chung (gutomitai@gmail.com)
"""

from keras.layers import Input, Dense
from keras.engine.input_layer import InputLayer

def reverse_model(model):
    """Reverse model.
    
    Shared layer, multiple nodes?
    
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
    output_layers = [layer for layer in layer if layer.name in output_layer_names]
    
    # Reconstruct the model reversely.
    outputs = model.outputs
    
    # About the first output tensor.
    output = outputs[0]
    
def _reverse_output(layer, tensor):
    """Reverse output layers recursively.
    
    Parameters
    ----------
    layer: Keras layer
        Layer instance.
    
    Returns
    -------
    Keras layer
        Reversed layer.
    """
    
    # Check exception.
    # TODO
    
    if isinstance(layer, Dense):
        # Get inbound nodes.
        i_nodes = layer._inbound_nodes
        
        # Create reverse layers.
        input1 = Input(shape=(layer.output_shape[1], ))
        r_dense_layer = Dense(layer.input_shape[1], activation=layer.activation) # Bias?
        x = r_dense_layer(input1)
        
        return r_dense_layer
    else:
        pass