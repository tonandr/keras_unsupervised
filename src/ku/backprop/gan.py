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
import pandas as pd

from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda
from keras.utils import multi_gpu_model
from keras import optimizers
from keras.models import Sequential
import keras.backend as K
from keras.engine.input_layer import InputLayer
from abc import abstractmethod

EPSILON = 1e-8

def gan_gen_loss(y_true, y_pred):
    return 1.0 * K.log(1.0 - y_pred + EPSILON) #?

def gan_disc_loss(y_true, y_pred):
    return 1.0 * K.log(y_pred + EPSILON) #?

class AbstractGAN(object):
    """Abstract generative adversarial network."""
    
    def __init__(self, hps, nn_arch):
        self.hps = hps
        self.nn_arch = nn_arch
    
    @abstractmethod
    def _create_generator(self):
        """Create the generator."""
    
    @abstractmethod        
    def _create_discriminator(self):
        """Create the discriminator."""
    
    def compile(self):
        """Create the GAM model and compile it."""
        
        # Check exception?
        
        # Design gan according to input and output nodes for each model.
        # Generator inputs.
        gen_layers = self.gen.layers()
        gen_input_layers = [layer for layer in gen_layers if isinstance(layer, InputLayer) == True]
        gen_output_layer_names = [t.name.split('/')[0] for t in self.gen.outputs]
        gen_output_layers = [layer for layer in layer if layer.name in gen_output_layer_names]        
        
        z_inputs = [Input(shape=self.layer.input_shape[1:]) for layer in gen_input_layers]
        
        # Discriminator inputs.
        disc_layers = self.disc.layers()
        disc_input_layers = [layer for layer in disc_layers if isinstance(layer, InputLayer) == True]
        disc_output_layer_names = [t.name.split('/')[0] for t in self.disc.outputs]
        disc_output_layers = [layer for layer in layer if layer.name in disc_output_layer_names]        
        
        x_inputs = [Input(shape=self.layer.input_shape[1:]) for layer in disc_input_layers]        
        
        z_outputs = self.gen(z_inputs)
        z_p_outputs = self.dist(z_outputs)
        x_outputs = self.disc(x_inputs)

        self.gan = Model(inputs=[z_inputs] + [x_inputs], outputs=[x_outputs] + [z_p_outputs])

        opt = optimizers.Adam(lr=self.hps['lr']
                                    , beta_1=self.hps['beta_1']
                                    , beta_2=self.hps['beta_2']
                                    , decay=self.hps['decay'])
        
        # Losses.
        disc_losses = [gan_disc_loss for _ in len(x_outputs)]
        gen_losses = [gan_gen_loss for _ in len(z_p_outputs)]
        
        self.gan.compile(optimizer=opt
                         , loss=disc_losses + gen_losses
                         , loss_weights=[-1.0, -1.0])
            
    def fit(self, x_inputs, x_outputs):
        """Train the GAN model.
        
        Parameters
        ----------
        x_inputs : list.
            Training data numpy array list.
        x_outputs : list.
            Ground truth data numpy array list.
        """
        num_samples = x_inputs[0].shape[0]
        
        for e_i in self.hps['epochs']:
            # Create z_inputs, z_p_outputs, z_outputs.
            z_inputs = [np.random.rand(*list([num_samples] + list(self.gen.get_input_shape_at(i)[1:]))) \
                        for i in range(len(self.gen.inputs()))]
            z_p_outputs = [np.zeros(shape=tuple([num_samples] + list(self.disc.get_output_shape_at(i)[1:]))) \
                        for i in range(len(self.dist.outputs()))]
            z_outputs = [np.zeros(shape=tuple([num_samples] + list(self.gen.get_output_shape_at(i)[1:]))) \
                        for i in range(len(self.gen.outputs()))]  
            
            # Train gan.
            self.gan.fit(z_inputs + x_inputs
                         , x_outputs + z_p_outputs
                         , batch_size=self.hps['disc_batch_size']
                         , epochs=self.hps['disc_epochs']
                         , verbose=1) 
        
            # Train gen.
            self.gen.fit(z_inputs
                         , z_outputs
                         , batch_size=self.hps['gen_batch_size']
                         , epochs=self.hps['gen_epochs']
                         , verbose=1)
    
    