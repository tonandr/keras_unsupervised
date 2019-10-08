from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.models import Model, model_from_json
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.engine.input_layer import InputLayer

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

def save_model_jh5(model):
    """Save a model with the json model file and hdf5 weight data.
    
    Parameters
    ----------
    model: Keras model
        Model instance.
    """
    with open(model.name + ".json", 'w') as f:
        f.write(model.to_json())
    
    model.save_weights(model.name + '.h5')
    
def load_model_jh5(model_name):
    """Load a model with the json model file and hdf5 weight data.
    
    Parameters
    ----------
    model_name: string
        Model name.
    """
    with open(model_name + '.json', 'r') as f:
        model_json = f.read()
        
    model = model_from_json(model_json)
    model.load_weights(model_name + '.h5')
    
    return model