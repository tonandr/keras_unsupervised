"""
Created on 2019. 6. 17.

@author: Inwoo Chung (gutomitai@gmail.com)
License: BSD 3 clause.

Revision:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from keras.models import Model
from keras.layers import Input, Dense
from keras.engine.input_layer import InputLayer
from keras import optimizers

from ..utility import reverse_undirected_model

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