"""
Created on 2019. 6. 17.

@author: Inwoo Chung (gutomitai@gmail.com)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC, abstractmethod
import warnings

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

class AbstractGAN(ABC):
    """Abstract generative adversarial network."""
    
    # Constants.
    GAN_PATH = 'style_gan_model.h5'
    DISC_EXT_PATH = 'style_gan_disc_ext_model.h5'    
    
    def __init__(self, conf):
        """
        Parameters
        ----------
        conf: dict
            Configuration.
        """
        self.conf = conf
        self.raw_data_path = self.conf['raw_data_path']
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']

        if self.conf['model_loading']: 
            if self.conf['multi_gpu']:
                # gan.
                self.gan = load_model(self.GAN_PATH, custom_objects={'gan_loss': gan_loss
                                                                       , 'disc_ext_loss': disc_ext_loss
                                                                       , 'disc_ext_loss2': disc_ext_loss2})
                self.gan_p = multi_gpu_model(self.gan, gpus=self.conf['num_gpus'])
                
                opt = optimizers.Adam(lr=self.hps['lr']
                                        , beta_1=self.hps['beta_1']
                                        , beta_2=self.hps['beta_2']
                                        , decay=self.hps['decay']) 
                self.gan_p.compile(optimizer=opt, loss=self.gan.losses) #?
                
                # disc_ext.
                self.disc_ext = load_model(self.DISC_EXT_PATH, custom_objects={'gan_loss': gan_loss
                                                                       , 'disc_ext_loss': disc_ext_loss
                                                                       , 'disc_ext_loss2': disc_ext_loss2})
                self.disc_ext_p = multi_gpu_model(self.gan, gpus=self.conf['num_gpus'])
                self.disc_ext_p.compile(optimizer=opt, loss=self.disc_ext.losses) #?
                
                # gen.
                self.gen = self.gan.get_layer('gen')
                self.gen_p = multi_gpu_model(self.gen, gpus=self.conf['num_gpus'])                                
            else:
                # gan.
                self.gan = load_model(self.GAN_PATH, custom_objects={'gan_loss': gan_loss
                                                                       , 'disc_ext_loss': disc_ext_loss
                                                                       , 'disc_ext_loss2': disc_ext_loss2})
                # disc_ext.
                self.disc_ext = load_model(self.DISC_EXT_PATH, custom_objects={'gan_loss': gan_loss
                                                                       , 'disc_ext_loss': disc_ext_loss
                                                                       , 'disc_ext_loss2': disc_ext_loss2})
                # gen.
                self.gen = self.gan.get_layer('gen')
                    
    @abstractmethod
    def _create_generator(self):
        """Create the generator."""
        pass
    
    @abstractmethod        
    def _create_discriminator(self):
        """Create the discriminator."""
        pass
    
    def compile(self):
        """Create the GAN model and compile it."""
        
        # Check exception?
        if hasattr(self, 'gen') != True or hasattr(self, 'disc') != True:
            raise ValueError('The generator and discriminator must be created')
        
        # Design gan according to input and output nodes for each model.        
        # Design and compile disc_ext.
        x_inputs = self.disc.inputs  
        x_outputs = self.disc(x_inputs)
        z_inputs = self.gen.inputs
        
        if self.conf['multi_gpu']:
            self.gen_p = multi_gpu_model(self.gen, gpus=self.conf['num_gpus'])
               
        self.gen.trainable = False #?
        self.gen.name ='gen'    
        z_outputs = self.gen(z_inputs)
        self.disc.name = 'disc'
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
        
        if self.conf['multi_gpu']:
            self.disc_ext_p = multi_gpu_model(self.gan, gpus=self.conf['num_gpus'])
            self.disc_ext_p.compile(optimizer=opt, loss=self.disc_ext.losses) #?                     
               
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

        if self.conf['multi_gpu']:
            self.gan_p = multi_gpu_model(self.gan, gpus=self.conf['num_gpus'])
            self.gan_p.compile(optimizer=opt, loss=self.gan.losses) #?
                
        # Save models.
        self.disc_ext.save(self.DISC_EXT_PATH)
        self.gan.save(self.GAN_PATH)
            
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
        
        for e_i in range(self.hps['epochs']):
            for s_i in range(self.hps['batch_step']):
                for k_i in range(self.hps['disc_k_step']):
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
                    if self.conf['multi_gpu']:
                        self.disc_ext_p.train_on_batch(x_inputs_b + z_inputs_b
                                 , x_outputs_b + x2_outputs_b, verbose=1) #?                    
                    else:
                        self.disc_ext.train_on_batch(x_inputs_b + z_inputs_b
                                 , x_outputs_b + x2_outputs_b, verbose=1) #?
        
                z_inputs_b = [np.random.rand(*list([num_samples] + list(self.gen.get_input_shape_at(i)[1:]))) \
                                for i in range(len(self.gen.inputs()))]
                z_p_outputs_b = [np.zeros(shape=tuple([num_samples] + list(self.disc.get_output_shape_at(i)[1:]))) \
                                for i in range(len(self.disc.outputs()))]

                # Train gan.
                if self.conf['multi_gpu']:
                    self.gan_p.train_on_batch(z_inputs_b, z_p_outputs_b, verbose=1)
                else:
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
        generator: Generator
            Training data generator.
        max_queue_size: Integer
            Maximum size for the generator queue (default: 10).
        workers: Integer
            Maximum number of processes to get samples (default: 1, 0: main thread).
        use_multiprocessing: Boolean
            Multi-processing flag (default: False).
        shuffle: Boolean
            Batch shuffling flag (default: True).
        """
        
        # Check exception.?        
        # Get the output generator.
        if workers > 0:
            if isinstance(generator, Sequence):
                enq = OrderedEnqueuer(generator
                                  , use_multiprocessing=use_multiprocessing
                                  , shuffle=shuffle)
            else:
                enq = GeneratorEnqueuer(Generator
                                        , use_multiprocessing=use_multiprocessing)
                
            enq.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enq.get()
        else:
            if isinstance(generator, Sequence):
                output_generator = iter_sequence_infinite(generator)
            else:
                output_generator = generator
        
        # Train.        
        num_samples = self.hps['mini_batch_size']
        
        for e_i in range(self.hps['epochs']):
            for s_i in range(self.hps['batch_step']):
                for k_i in range(self.hps['disc_k_step']): #?
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
                    if self.conf['multi_gpu']:
                        self.disc_ext_p.train_on_batch(x_inputs_b + z_inputs_b
                                 , x_outputs_b + x2_outputs_b, verbose=1) #?                    
                    else:
                        self.disc_ext.train_on_batch(x_inputs_b + z_inputs_b
                                 , x_outputs_b + x2_outputs_b, verbose=1) #?
        
                z_inputs_b = [np.random.rand(*list([num_samples] + list(self.gen.get_input_shape_at(i)[1:]))) \
                                for i in range(len(self.gen.inputs()))]
                z_p_outputs_b = [np.zeros(shape=tuple([num_samples] + list(self.disc.get_output_shape_at(i)[1:]))) \
                                for i in range(len(self.disc.outputs()))]
                
                # Train gan.
                if self.conf['multi_gpu']:
                    self.gan_p.train_on_batch(z_inputs_b, z_p_outputs_b, verbose=1)
                else:
                    self.gan.train_on_batch(z_inputs_b, z_p_outputs_b, verbose=1)
 
    @abstractmethod    
    def generate(self, *args, **kwargs):
        """Generate styled images."""
        pass 