from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import model_from_json
     
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