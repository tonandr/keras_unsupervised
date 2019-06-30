"""
Created on 2019. 6. 17.

@author: Inwoo Chung (gutomitai@gmail.com)
License: BSD 3 clause.

Revision:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod

import numpy as np
import pandas as pd

from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda
from keras.utils import multi_gpu_model
from keras import optimizers
from keras.models import Sequential
import keras.backend as K
from keras.engine.input_layer import InputLayer
from keras.utils import Sequence, GeneratorEnqueuer, OrderedEnqueuer
from keras.engine.training_utils import iter_sequence_infinite
from _collections_abc import generator, Generator

EPSILON = 1e-8

def gan_loss(y_true, y_pred):
    return 1.0 * K.log(1.0 - y_pred + EPSILON) #?

def disc_ext_loss(y_true, y_pred):
    return 1.0 * K.log(y_pred + EPSILON) #?

def disc_ext_loss2(y_true, y_pred):
    return 1.0 * K.log(1.0 - y_pred + EPSILON) #?

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
        # TODO
        
        # Design gan according to input and output nodes for each model.        
        # Design and compile disc_ext.
        x_inputs = self.disc.inputs  
        x_outputs = self.disc(x_inputs)
        z_inputs = self.gen.inputs
        
        self.gen.trainable = False    
        z_outputs = self.gen(z_inputs)
        x2_outputs = self.disc(z_outputs)
        
        self.disc_ext = Model(inputs=x_inputs + z_inputs, outputs=x_outputs + x2_outputs)        
    
        opt = optimizers.Adam(lr=self.hps['lr']
                                    , beta_1=self.hps['beta_1']
                                    , beta_2=self.hps['beta_2']
                                    , decay=self.hps['decay'])

        disc_ext_losses1 = [disc_ext_loss for _ in len(x_outputs)] #?
        disc_ext_losses2 = [disc_ext_loss2 for _ in len(x2_outputs)] #?        
        self.dist_ext.compile(optimizer=opt
                         , loss=disc_ext_losses1 + disc_ext_losses2 
                         , loss_weights=-1.0)        
               
        # Design and compile gan.
        z_inputs = self.gen.inputs    
        z_outputs = self.gen(z_inputs)
        self.dist.trainable = False #?
        z_p_outputs = self.dist(z_outputs) #?

        self.gan = Model(inputs=[z_inputs], outputs=[z_p_outputs])
        gan_losses = [gan_loss for _ in len(z_p_outputs)]
        self.gan.compile(optimizer=opt
                         , loss=gan_losses #?
                         , loss_weights=1.0)
            
    def fit(self, x_inputs, x_outputs):
        """Train the GAN model.
        
        Parameters
        ----------
        x_inputs : list.
            Training data numpy array list.
        x_outputs : list.
            Ground truth data numpy array list.
        """
        num_samples = self.hps['mini_batch_size']
        
        for e_i in self.hps['epochs']:
            for s_i in self.hps['batch_step']:
                for k_i in self.hps['disc_k_step']:
                    # Create x_inputs_b, x_outputs_b, z_inputs_b, x2_outputs_b, z_p_outputs_b, z_outputs_b.
                    x_inputs_b = [x_inputs[i][np.random.rand(0, x_inputs[i].shape[0], num_samples)] \
                                  for i in range(len(x_inputs))]
                    x_outputs_b = [x_outputs[i][np.random.rand(0, x_outputs[i].shape[0], num_samples)] \
                                   for i in range(len(x_outputs))]
                    
                    z_inputs_b = [np.random.rand(*list([num_samples] + list(self.gen.get_input_shape_at(i)[1:]))) \
                                for i in range(len(self.gen.inputs()))]
                    x2_outputs_b = [np.zeros(shape=tuple([num_samples] + list(self.disc.get_output_shape_at(i)[1:]))) \
                                for i in range(len(self.disc.outputs()))]
         
                    # Train disc.
                    self.disc.train_on_batch(x_inputs_b + z_inputs_b
                                 , x_outputs_b + x2_outputs_b, verbose=1) #?
        
                z_inputs_b = [np.random.rand(*list([num_samples] + list(self.gen.get_input_shape_at(i)[1:]))) \
                                for i in range(len(self.gen.inputs()))]
                z_p_outputs_b = [np.zeros(shape=tuple([num_samples] + list(self.disc.get_output_shape_at(i)[1:]))) \
                                for i in range(len(self.disc.outputs()))]
                # Train gan.
                self.gan.train_on_batch(z_inputs_b, z_p_outputs_b, verbose=1)

    def fit_generator(self
                      , generator
                      , max_queue_size=10
                      , workers=1
                      , use_multiprocessing=False
                      , shuffle=True):
        """Train the GAN model with the generator.
        
        Parameters
        ----------
        generator : Generator.
            Training data generator.
        ?
        """
        
        # Check exception.
        # TODO
        
        # Get the output generator.
        if workers > 0:
            if isinstance(generator, Sequence):
                enq = OrderedEnqueuer(generator
                                  , use_multiprocessing=use_multiprocessing
                                  , shuffle=shuffle)
            else:
                enq = GeneratorEnqueuer(Generator
                                        , use_multiprocessing=use_multiprocessing)
            
            output_generator = enq.get()
        else:
            if isinstance(generator, Sequence):
                output_generator = iter_sequence_infinite(generator)
            else:
                output_generator = generator
        
        # Train.        
        num_samples = self.hps['mini_batch_size']
        
        for e_i in self.hps['epochs']:
            for s_i in self.hps['batch_step']:
                for k_i in self.hps['disc_k_step']: #?
                    x_inputs, x_outputs = next(output_generator)
                    
                    # Create x_inputs_b, x_outputs_b, z_inputs_b, x2_outputs_b, z_p_outputs_b, z_outputs_b.
                    x_inputs_b = [x_inputs[i][np.random.rand(0, x_inputs[i].shape[0], num_samples)] \
                                  for i in range(len(x_inputs))]
                    x_outputs_b = [x_outputs[i][np.random.rand(0, x_outputs[i].shape[0], num_samples)] \
                                   for i in range(len(x_outputs))]
                    
                    z_inputs_b = [np.random.rand(*list([num_samples] + list(self.gen.get_input_shape_at(i)[1:]))) \
                                for i in range(len(self.gen.inputs()))]
                    x2_outputs_b = [np.zeros(shape=tuple([num_samples] + list(self.disc.get_output_shape_at(i)[1:]))) \
                                for i in range(len(self.disc.outputs()))]
         
                    # Train disc.
                    self.disc.train_on_batch(x_inputs_b + z_inputs_b
                                 , x_outputs_b + x2_outputs_b, verbose=1) #?
        
                z_inputs_b = [np.random.rand(*list([num_samples] + list(self.gen.get_input_shape_at(i)[1:]))) \
                                for i in range(len(self.gen.inputs()))]
                z_p_outputs_b = [np.zeros(shape=tuple([num_samples] + list(self.disc.get_output_shape_at(i)[1:]))) \
                                for i in range(len(self.disc.outputs()))]
                # Train gan.
                self.gan.train_on_batch(z_inputs_b, z_p_outputs_b, verbose=1)
     