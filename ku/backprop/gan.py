from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC, abstractmethod
import os
import warnings
import shutil

import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K 
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.utils import Sequence, GeneratorEnqueuer, OrderedEnqueuer
from tensorflow.keras.callbacks import TensorBoard

from tensorflow_core.python.keras.utils.generic_utils import to_list, CustomObjectScope
from tensorflow_core.python.keras.utils.data_utils import iter_sequence_infinite
from tensorflow_core.python.keras import callbacks as cbks
from tensorflow_core.python.keras.engine import training_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys #?

from ..engine_ext import ModelExt
from ..layer_ext import InputRandomUniform
from ..loss_ext import DiscExtRegularLoss1, DiscExtRegularLoss2, GenDiscRegularLoss2
from ..loss_ext import GenDiscWGANLoss, DiscExtWGANLoss, DiscExtWGANGPLoss
from ..loss_ext import SoftPlusNonSatLoss, SoftPlusLoss, RPenaltyLoss

# GAN mode.
STYLE_GAN = 0
PIX2PIX_GAN = 1

# Loss configuration type.
LOSS_CONF_TYPE_REGULAR = 0
LOSS_CONF_TYPE_WGAN_GP = 1
LOSS_CONF_TYPE_NON_SAT_R1 = 2

def get_loss_conf(hps, lc_type, *args, **kwargs):
    """Get the GAN loss configuration.
    
    Parameters
    ----------
    hps: Dict.
        GAN model's hyper-parameters.
    lc_type: Integer.
        Loss configuration type.
    
    Returns
    -------
        Loss configuration.
            Dict.
    """
    loss_conf = {}
    if lc_type == LOSS_CONF_TYPE_REGULAR:
        loss_conf = {'disc_ext_losses': [DiscExtRegularLoss1(), DiscExtRegularLoss2()]
                    , 'disc_ext_loss_weights': [-1.0, -1.0]
                    , 'gen_disc_losses': [GenDiscRegularLoss2()]
                    , 'gen_disc_loss_weights': [-1.0]}
    elif lc_type == LOSS_CONF_TYPE_WGAN_GP:
        loss_conf = {'disc_ext_losses': [DiscExtWGANLoss()
                                , DiscExtWGANLoss()
                                , DiscExtWGANGPLoss(input_variables=kwargs['wgan_gp_input_variables'] #?
                                                        , wgan_lambda=hps['wgan_lambda']
                                                        , wgan_target=hps['wgan_target'])]
                    , 'disc_ext_loss_weights': [-1.0, 1.0, 1.0]
                    , 'gen_disc_losses': [DiscExtWGANLoss()]
                    , 'gen_disc_loss_weights': [-1.0]}
    elif lc_type == LOSS_CONF_TYPE_NON_SAT_R1:
        loss_conf = {'disc_ext_losses': [SoftPlusNonSatLoss(name='real_loss')
                                , RPenaltyLoss(name='r_penalty_loss'
                                               , model=kwargs['disc_ext']
                                               , input_variable_orders=[0] 
                                               , r_gamma=hps['r_gamma'])
                                , SoftPlusLoss(name='fake_loss')]
                    , 'disc_ext_loss_weights': [1.0, 1.0, 1.0]
                    , 'gen_disc_losses': [SoftPlusNonSatLoss()]
                    , 'gen_disc_loss_weights': [1.0]}
    else:
        raise ValueError('type is not valid.')

    return loss_conf

class AbstractGAN(ABC):
    """Abstract generative adversarial network."""

    # Constants.
    GEN_DISC_PATH = 'gen_disc.h5'
    DISC_EXT_PATH = 'disc_ext.h5'
    
    def __init__(self, conf):
        """
        Parameters
        ----------
        conf: dict
            Configuration.
        """
        self._is_gan_compiled = False
        self.conf = conf #?
        
        if self.conf['model_loading']:
            if not hasattr(self, 'custom_objects'):
                RuntimeError('Before models, custom_objects must be created.')
            
            self.custom_objects['GenDiscRegularLoss2'] = GenDiscRegularLoss2
            self.custom_objects['GenDiscWGANLoss'] = GenDiscWGANLoss
            self.custom_objects['ModelExt'] = ModelExt
            self.custom_objects['SoftPlusNonSatLoss'] = SoftPlusNonSatLoss
                                                        
            with CustomObjectScope(self.custom_objects):
                # disc_ext.
                self.disc_ext_init = load_model(self.DISC_EXT_PATH, compile=True) #?
                 
                # gen_disc.
                self.gen_disc_init = load_model(self.GEN_DISC_PATH, compile=True) #?
                                        
                # gen.
                self.gen = self.gen_disc_init.get_layer('gen')
                
                if conf['multi_gpu']:
                    self.disc_ext = multi_gpu_model(self.disc_ext_init, gpus=self.conf['num_gpus'])
                    self.disc_ext.compile(optimizer=self.disc_ext_init.optimizer
                                          , loss=self.disc_ext_init.losses
                                          , loss_weights=self.disc_ext_init.loss_weights
                                          , run_eagerly=True)
                    
                    self.gen_disc = multi_gpu_model(self.gen_disc_init, gpus=self.conf['num_gpus'])
                    self.gen_disc.compile(optimizer=self.gen_disc_init.optimizer
                                          , loss=self.gen_disc_init.losses
                                          , loss_weights=self.gen_disc_init.loss_weights
                                          , run_eagerly=True)
                    
                    self.gen_p = multi_gpu_model(self.gen, gpus=self.conf['num_gpus'])

            self._is_gan_compiled = True        
    
    @property
    def is_gan_compiled(self):
        return self._is_gan_compiled
                                    
    @abstractmethod
    def _create_generator(self):
        """Create the generator."""
        raise NotImplementedError('_crate_generator is not implemented.')
    
    @abstractmethod        
    def _create_discriminator(self):
        """Create the discriminator."""
        raise NotImplementedError('_crate_discriminator is not implemented.')
    
    def compose_gan(self):
        """Compose the GAN model."""
        raise NotImplementedError('compose_gan is not implemented.')
    
    def create_compose_gan_with_mode_func(self, mode):
        """Create the compose_gan_with_mode function.
        
        Parameters
        ----------
        mode: Integer.
            GAN model composing mode.
        """
        self.compose_gan = compose_gan_with_mode(self, mode)
        
    def compile(self
                , disc_ext_opt
                , disc_ext_losses
                , disc_ext_loss_weights
                , gen_disc_opt
                , gen_disc_losses
                , gen_disc_loss_weights
                , disc_ext_metrics=None
                , gen_disc_metrics=None):
        """compile."""
        
        # Check exception.
        assert hasattr(self, 'disc_ext') and hasattr(self, 'gen_disc')

        self.disc_ext.compile(optimizer=disc_ext_opt
                         , loss=disc_ext_losses
                         , loss_weights=disc_ext_loss_weights
                         , metrics=disc_ext_metrics
                         , run_eagerly=True)
        self.gen_disc.compile(optimizer=gen_disc_opt
                         , loss=gen_disc_losses
                         , loss_weights=gen_disc_loss_weights
                         , metrics=gen_disc_metrics
                         , run_eagerly=True)
        self._is_gan_compiled = True

    def _compile_wgan_gp(self):
        """Compile wgan_gp."""
        
        # Design gan according to input and output nodes for each model.        
        # Design and compile disc_ext.
        # x.
        disc_inputs = self.disc.inputs if self.nn_arch['label_usage'] else [self.disc.inputs]
        x_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in disc_inputs]  
        x_outputs = [self.disc(x_inputs)]
        
        # x_tilda.
        gen_inputs = self.gen.inputs
        z_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in gen_inputs] 
        
        if self.conf['multi_gpu']:
            self.gen_p = multi_gpu_model(self.gen, gpus=self.conf['num_gpus'])
               
        self.gen.trainable = False
        for layer in self.gen.layers: layer.trainable = False
        z_outputs = self.gen(z_inputs) if self.nn_arch['label_usage'] else [self.gen(z_inputs)]
        self.disc.trainable = True
        for layer in self.disc.layers: layer.trainable = True 
        x2_outputs = [self.disc(z_outputs)]
        
        # x_hat.
        x3_inputs = [Input(shape=(1, ))] # Trivial input.
        x3_epsilon = [InputRandomUniform((1, 1, 1))(x3_inputs)]
        x3 = Lambda(lambda x: x[2] * x[1] + (1.0 - x[2]) * x[0])([x_inputs[0]] + [z_outputs[0]] + x3_epsilon)
        x3_outputs = [self.disc([x3, x_inputs[1]])]
        
        self.disc_ext = ModelExt(inputs=x_inputs + z_inputs + x3_inputs
                              , outputs=x_outputs + x2_outputs + x3_outputs)        
    
        opt = optimizers.Adam(lr=self.hps['lr']
                                    , beta_1=self.hps['beta_1']
                                    , beta_2=self.hps['beta_2']
                                    , decay=self.hps['decay'])

        # Make losses.        
        self.disc_ext_losses = [DiscExtWGANLoss()
                                , DiscExtWGANLoss()
                                , DiscExtWGANGPLoss(input_variables=x3
                                                        , wgan_lambda=self.conf['hps']['wgan_lambda']
                                                        , wgan_target=self.conf['hps']['wgan_target'])]
        self.disc_ext_loss_weights = [-1.0, 1.0, 1.0]

        if hasattr(self, 'disc_ext_losses') == False \
            or hasattr(self, 'disc_ext_loss_weights') == False:
            raise RuntimeError("disc_ext_losses and disc_ext_weights must be created.")
                    
        self.disc_ext.compile(optimizer=opt
                         , loss=self.disc_ext.losses
                         , loss_weights=self.disc_ext.loss_weights
                         , run_eagerly=True)
        
        if self.conf['multi_gpu']:
            self.disc_ext_p = multi_gpu_model(self.disc_ext, gpus=self.conf['num_gpus'])
            self.disc_ext_p.compile(optimizer=opt
                                    , loss=self.disc_ext.losses
                                    , loss_weights=self.disc_ext.loss_weights
                                    , run_eagerly=True)                     
               
        # Design and compile gen_disc.
        gen_inputs = self.gen.inputs
        z_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in gen_inputs] 
        self.gen.trainable = True
        for layer in self.gen.layers: layer.trainable = True
        z_outputs = self.gen(z_inputs) if self.nn_arch['label_usage'] else [self.gen(z_inputs)]
        self.disc.trainable = False
        for layer in self.disc.layers: layer.trainable = False
        z_p_outputs = [self.disc(z_outputs)]

        self.gen_disc = ModelExt(inputs=z_inputs, outputs=z_p_outputs)
        
        # Make losses.
        self.gen_disc_losses = [DiscExtWGANLoss()]
        self.gen_disc_loss_weights = [-1.0]
        
        if hasattr(self, 'gen_disc_losses') == False \
            or hasattr(self, 'gen_disc_loss_weights') == False:
            raise RuntimeError("gen_disc_losses and gen_disc_weights must be created.")
                
        self.gen_disc.compile(optimizer=opt
                         , loss=self.gen_disc.losses
                         , loss_weights=self.gen_disc.loss_weights
                         , run_eagerly=True)

        if self.conf['multi_gpu']:
            self.gen_disc_p = multi_gpu_model(self.gen_disc, gpus=self.conf['num_gpus'])
            self.gen_disc_p.compile(optimizer=opt
                         , loss=self.gen_disc.losses
                         , loss_weights=self.gen_disc.loss_weights
                         , run_eagerly=True)

    def _compile_gan_with_non_sat_rl_loss(self):
        """Compile gan with non-saturation and r1 loss."""
        
        # Design gan according to input and output nodes for each model.        
        # Design and compile disc_ext.
        # x.
        disc_inputs = self.disc.inputs if self.nn_arch['label_usage'] else [self.disc.inputs]
        x_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in disc_inputs] 
        x_outputs = [self.disc(x_inputs)]
        
        # x_tilda.
        gen_inputs = self.gen.inputs
        z_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in gen_inputs]  
        
        if self.conf['multi_gpu']:
            self.gen_p = multi_gpu_model(self.gen, gpus=self.conf['num_gpus'])
                
        self.gen.trainable = False
        for layer in self.gen.layers: layer.trainable = False
        z_outputs = self.gen(z_inputs) if self.nn_arch['label_usage'] else [self.gen(z_inputs)]

        self.disc.trainable = True
        for layer in self.disc.layers: layer.trainable = True
        x2_outputs = [self.disc(z_outputs)]
                
        self.disc_ext = ModelExt(inputs=x_inputs + z_inputs
                              , outputs=x_outputs + x_outputs + x2_outputs)        
    
        opt = optimizers.Adam(lr=self.hps['lr']
                                    , beta_1=self.hps['beta_1']
                                    , beta_2=self.hps['beta_2']
                                    , decay=self.hps['decay'])

        # Make losses.        
        self.disc_ext_losses = [SoftPlusNonSatLoss(name='real_loss')
                                , RPenaltyLoss(name='r_penalty_loss'
                                               , model = self.disc_ext
                                               , input_variable_orders = [0] 
                                               , r_gamma=self.conf['hps']['r_gamma'])
                                , SoftPlusLoss(name='fake_loss')]
        self.disc_ext_loss_weights = [1.0, 1.0, 1.0]

        if hasattr(self, 'disc_ext_losses') == False \
            or hasattr(self, 'disc_ext_loss_weights') == False:
            raise RuntimeError("disc_ext_losses and disc_ext_weights must be created.")
                    
        self.disc_ext.compile(optimizer=opt
                         , loss=self.disc_ext_losses
                         , loss_weights=self.disc_ext_loss_weights
                         , run_eagerly=True)
        
        if self.conf['multi_gpu']:
            self.disc_ext_p = multi_gpu_model(self.disc_ext, gpus=self.conf['num_gpus'])
            self.disc_ext_p.compile(optimizer=opt
                                    , loss=self.disc_ext.losses
                                    , loss_weights=self.disc_ext_loss_weights
                                    , run_eagerly=True)                    
               
        # Design and compile gen_disc.
        gen_inputs = self.gen.inputs
        z_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in gen_inputs] 
        self.gen.trainable = True
        for layer in self.gen.layers: layer.trainable = True     
        z_outputs = self.gen(z_inputs) if self.nn_arch['label_usage'] else [self.gen(z_inputs)]
        
        self.disc.trainable = False
        for layer in self.disc.layers: layer.trainable = False
        z_p_outputs = [self.disc(z_outputs)]

        self.gen_disc = ModelExt(inputs=z_inputs, outputs=z_p_outputs)
        
        # Make losses.
        self.gen_disc_losses = [SoftPlusNonSatLoss()]        
        self.gen_disc_loss_weights = [1.0]
        
        if hasattr(self, 'gen_disc_losses') == False \
            or hasattr(self, 'gen_disc_loss_weights') == False:
            raise RuntimeError("gen_disc_losses and gen_disc_weights must be created.")
                
        self.gen_disc.compile(optimizer=opt
                         , loss=self.gen_disc_losses #?
                         , loss_weights=self.gen_disc_loss_weights
                         , run_eagerly=True)

        if self.conf['multi_gpu']:
            self.gen_disc_p = multi_gpu_model(self.gen_disc, gpus=self.conf['num_gpus'])
            self.gen_disc_p.compile(optimizer=opt
                         , loss=self.gen_disc_losses
                         , loss_weights=self.gen_disc_loss_weights
                         , run_eagerly=True)                
    
    @abstractmethod
    def gen_disc_ext_data_fun(self, generator, *args, **kwargs):
        """Generate disc_ext data.
        
        Parameters
        ----------
        generator: Generator.
            Data generator.
        """
        raise NotImplementedError('gen_disc_ext_data_fun is not implemented.')

    @abstractmethod
    def gen_gen_disc_data_fun(self, generator, *args, **kwargs):
        """Generate gen_disc data.
        
        Parameters
        ----------
        generator: Generator.
            Data generator.
        """
        raise NotImplementedError('gen_gen_disc_data_fun is not implemented.')        
    
    def fit_generator(self
                      , generator
                      , gen_disc_ext_data_fun
                      , gen_gen_disc_data_fun
                      , verbose=1
                      , callbacks_disc_ext=None
                      , callbacks_gen_disc=None
                      , validation_data_gen=None #?
                      , validation_steps=None
                      , validation_freq=1 #?
                      , class_weight=None #?
                      , max_queue_size=10
                      , workers=1
                      , use_multiprocessing=False #?
                      , shuffle=True
                      , initial_epoch=0
                      , save_f=True): #?
        """Train the GAN model with the generator.
        
        Parameters
        ----------
        generator: Generator
            Training data generator.
        verbose: Integer 
            Verbose mode (default=1).
        callback_disc_ext: list
            disc_ext callbacks (default=None).
        callback_gen_disc: list
            gen_disc callbacks (default=None).
        validation_data_gen: Generator or Sequence
            Validation generator or sequence (default=None).
        validation_steps: Integer
            Validation steps (default=None).
        validation_freq: Integer
            Validation frequency (default=1).
        max_queue_size: Integer
            Maximum size for the generator queue (default: 10).
        workers: Integer
            Maximum number of processes to get samples (default: 1, 0: main thread).
        use_multiprocessing: Boolean
            Multi-processing flag (default: False).
        shuffle: Boolean
            Batch shuffling flag (default: True).
        initial_epoch: Integer
            Initial epoch (default: 0).
        
        Returns
        -------
        Training history.
            tuple
        """
        
        # Check exception.
        do_validation = bool(validation_data_gen)
        if do_validation:
            assert hasattr(validation_data_gen, 'next') or \
                hasattr(validation_data_gen, '__next') or \
                isinstance(validation_data_gen, Sequence)
            
            if not isinstance(validation_data_gen, Sequence):
                assert validation_steps #?
        
            assert isinstance(validation_freq, int)
                    
        if not isinstance(generator, Sequence) and use_multiprocessing and workers > 1:
            warnings.warn(UserWarning('For multiprocessing, use the instance of Sequence.'))

        # Initialize the results directory
        if not os.path.isdir(os.path.join('results')):
            os.mkdir(os.path.join('results'))
        else:
            shutil.rmtree(os.path.join('results'))
            os.mkdir(os.path.join('results'))
            
        try:                    
            # Get the validation generator and output generator.
            if do_validation:
                if workers > 0:
                    if isinstance(validation_data_gen, Sequence):
                        val_enq = OrderedEnqueuer(validation_data_gen
                                                  , use_multiprocessing=use_multiprocessing) # shuffle?
                        validation_steps = validation_steps or len(validation_data_gen)
                    else:
                        val_enq = GeneratorEnqueuer(validation_data_gen
                                                    , use_multiprocessing=use_multiprocessing)
                    
                    val_enq.start(workers=workers, max_queue_size=max_queue_size)
                    val_generator = val_enq.get()
                else:
                    if isinstance(validation_data_gen, Sequence):
                        val_generator = iter_sequence_infinite(validation_data_gen)
                        validation_steps = validation_steps or len(validation_data_gen)
                    else:
                        val_generator = validation_data_gen 
            
            if workers > 0:
                if isinstance(generator, Sequence):
                    enq = OrderedEnqueuer(generator
                                      , use_multiprocessing=use_multiprocessing
                                      , shuffle=shuffle)
                else:
                    enq = GeneratorEnqueuer(generator
                                            , use_multiprocessing=use_multiprocessing)
                    
                enq.start(workers=workers, max_queue_size=max_queue_size)
                output_generator = enq.get()
            else:
                if isinstance(generator, Sequence):
                    output_generator = iter_sequence_infinite(generator)
                else:
                    output_generator = generator
                 
            # Callbacks.            
            # disc_ext.
            #out_labels_disc_ext = self.disc_ext.metrics_names if hasattr(self.disc_ext, 'metrics_names') else []
            out_labels_disc_ext = ['loss'] + [v.name for v in self.disc_ext.loss_functions]
            callback_metrics_disc_ext = out_labels_disc_ext \
                + ['val_' + out_label for out_label in out_labels_disc_ext]
            self.disc_ext.history = cbks.History()
            _callbacks = [cbks.BaseLogger(stateful_metrics=[])]
            if verbose:
                _callbacks.append(cbks.ProgbarLogger(count_mode='steps'
                                                     , stateful_metrics=[]))
                
            # Tensorboard callback.
            callback_tb = TensorBoard(log_dir='.\\logs'
                                           , histogram_freq=1
                                           , batch_size=self.hps['mini_batch_size']
                                           , write_graph=True
                                           , write_grads=True
                                           , write_images=True
                                           , update_freq='batch')
            _callbacks.append(callback_tb)
                            
            _callbacks += (callbacks_disc_ext or []) + [self.disc_ext.history]
            callbacks_disc_ext = cbks.configure_callbacks(_callbacks
                                                          , self.disc_ext
                                                          , do_validation=do_validation
                                                          , epochs=self.hps['epochs']
                                                          , steps_per_epoch=self.hps['batch_step'] * self.hps['disc_k_step']
                                                          , samples=self.hps['batch_step'] * self.hps['disc_k_step']
                                                          , verbose=1
                                                          , mode=ModeKeys.TRAIN)  
          
            # gen_disc.
            #out_labels_gen_disc = self.gen_disc.metrics_names if hasattr(self.gen_disc, 'metrics_names') else []
            out_labels_gen_disc = ['loss'] + [v.name for v in self.gen_disc.loss_functions]
            callback_metrics_gen_disc = out_labels_gen_disc \
                + ['val_' + out_label for out_label in out_labels_gen_disc]
            self.gen_disc.history = cbks.History()
            _callbacks = [cbks.BaseLogger(stateful_metrics=[])]
            if verbose:
                _callbacks.append(cbks.ProgbarLogger(count_mode='steps'
                                                     , stateful_metrics=[]))
            
            # Tensorboard callback.
            callback_tb = TensorBoard(log_dir='.\\logs'
                                           , histogram_freq=1
                                           , batch_size=self.hps['mini_batch_size']
                                           , write_graph=True
                                           , write_grads=True
                                           , write_images=True
                                           , update_freq='batch')
            _callbacks.append(callback_tb)
            
            _callbacks += (callbacks_gen_disc or []) + [self.gen_disc.history]
            callbacks_gen_disc = cbks.configure_callbacks(_callbacks
                                                          , self.gen_disc
                                                          , do_validation=do_validation
                                                          , epochs=self.hps['epochs']
                                                          , steps_per_epoch=self.hps['batch_step']
                                                          , samples=self.hps['batch_step']
                                                          , verbose=1
                                                          , mode=ModeKeys.TRAIN) 

            aggr_metrics_disc_ext = training_utils.MetricsAggregator(True, steps=self.hps['batch_step'])
            aggr_metrics_gen_disc = training_utils.MetricsAggregator(True, steps=self.hps['batch_step'])
            
            # Train.
            callbacks_disc_ext.model.stop_training = False
            callbacks_gen_disc.model.stop_training = False  
            callbacks_disc_ext._call_begin_hook(ModeKeys.TRAIN)
            callbacks_gen_disc._call_begin_hook(ModeKeys.TRAIN)
            
            initial_epoch = self.disc_ext._maybe_load_initial_epoch_from_ckpt(initial_epoch, ModeKeys.TRAIN) #?                                  
            
            for e_i in range(initial_epoch, self.hps['epochs']):
                if callbacks_disc_ext.model.stop_training or callbacks_gen_disc.model.stop_training:
                    break
                
                self.disc_ext.reset_metrics()
                self.gen_disc.reset_metrics()
                    
                epochs_log_disc_ext = {}
                epochs_log_gen_disc = {}
                                                   
                callbacks_disc_ext.on_epoch_begin(e_i, epochs_log_disc_ext)
                callbacks_gen_disc.on_epoch_begin(e_i, epochs_log_gen_disc)

                for s_i in range(self.hps['batch_step']):
                    for k_i in range(self.hps['disc_k_step']):
                        # Build batch logs.
                        k_batch_logs = {'batch': self.hps['disc_k_step'] * s_i + k_i + 1, 'size': self.hps['mini_batch_size']}
                        callbacks_disc_ext._call_batch_hook(ModeKeys.TRAIN
                                                            , 'begin'
                                                            , self.hps['disc_k_step'] * s_i + k_i + 1
                                                            , k_batch_logs)
                                                
                        inputs, outputs = gen_disc_ext_data_fun(output_generator)
                        outs = self.disc_ext.train_on_batch(inputs
                                 , outputs
                                 , class_weight=class_weight
                                 , reset_metrics=True) #?
                        del inputs, outputs
                        outs = to_list(outs) #?
                        
                        if s_i == 0:
                            aggr_metrics_disc_ext.create(outs)
                        
                        metrics_names = ['disc_ext_loss'] + [v.name for v in self.disc_ext.loss_functions]                             
                            
                        for l, o in zip(metrics_names, outs):
                            k_batch_logs[l] = o                        
                                    
                        callbacks_disc_ext._call_batch_hook(ModeKeys.TRAIN
                                                            , 'end'
                                                            , self.hps['disc_k_step'] * s_i + k_i + 1
                                                            , k_batch_logs)
                        print('\n', k_batch_logs)
                        
                    # Build batch logs.
                    batch_logs = {'batch': s_i + 1, 'size': self.hps['mini_batch_size']}
                    callbacks_gen_disc._call_batch_hook(ModeKeys.TRAIN
                                                        , 'begin'
                                                        , s_i
                                                        , batch_logs)

                    inputs, outputs = gen_gen_disc_data_fun(output_generator)
                    outs = self.gen_disc.train_on_batch(inputs
                                                        , outputs
                                                        , class_weight=class_weight
                                                        , reset_metrics=False)
                    del inputs, outputs
                    outs = to_list(outs)
                    
                    if s_i == 0:
                        aggr_metrics_gen_disc.create(outs)
                    
                    metrics_names = ['gen_disc_loss'] + [v.name for v in self.gen_disc.loss_functions]
                    
                    for l, o in zip(metrics_names, outs): #?
                        batch_logs[l] = o
        
                    callbacks_gen_disc._call_batch_hook(ModeKeys.TRAIN
                                                        , 'end'
                                                        , s_i
                                                        , batch_logs)
                    print('\n', batch_logs)
                
                aggr_metrics_disc_ext.finalize()
                aggr_metrics_gen_disc.finalize()
                
                # Make epochs log.
                outs_disc_ext = to_list(aggr_metrics_disc_ext.results)
                for out_label, out in zip(out_labels_disc_ext, outs_disc_ext):
                    epochs_log_disc_ext[out_label] = out
                                
                outs_gen_disc = to_list(aggr_metrics_gen_disc.results)
                for out_label, out in zip(out_labels_gen_disc, outs_gen_disc):
                    epochs_log_gen_disc[out_label] = out
                    
                # Do validation.
                if not do_validation: #?
                    if e_i % validation_freq == 0: #?
                        # disc_ext.
                        val_outs_disc_ext = self._evaluate_disc_ext(val_generator #?
                                                                      , gen_disc_ext_data_fun
                                                                      , callbacks=callbacks_disc_ext
                                                                      , workers=0)
                        
                        # gen_disc.
                        val_outs_gen_disc = self._evaluate_gen_disc(val_generator
                                                                      , gen_gen_disc_data_fun
                                                                      , callbacks=callbacks_gen_disc
                                                                      , workers=0)                                            
                        # Make epochs log.
                        val_outs_disc_ext = to_list(val_outs_disc_ext)
                        for out_label, val_out in zip(out_labels_disc_ext, val_outs_disc_ext):
                            epochs_log_disc_ext['val_' + out_label] = val_out
                                        
                        val_outs_gen_disc = to_list(val_outs_gen_disc)
                        for out_label, val_out in zip(out_labels_gen_disc, val_outs_gen_disc):
                            epochs_log_gen_disc['val_' + out_label] = val_out
                                                                
                callbacks_disc_ext.on_epoch_end(e_i, epochs_log_disc_ext)
                callbacks_gen_disc.on_epoch_end(e_i, epochs_log_gen_disc)
                
                if save_f:
                    self.save_gan_model()
                                
            self.disc_ext._successful_loop_finish = True
            self.gen_disc._successful_loop_finish = True
                
            callbacks_disc_ext._call_end_hook(ModeKeys.TRAIN)
            callbacks_gen_disc._call_end_hook(ModeKeys.TRAIN) 
        finally:
            try:
                if enq is not None:
                    enq.stop()
            finally:
                if val_enq is not None:
                    val_enq.stop()

        return self.disc_ext.history, self.gen_disc.history

    def fit_generator_progressively(self
                      , generator
                      , fixed_layer_names
                      , gen_disc_ext_data_fun
                      , gen_gen_disc_data_fun
                      , verbose=1
                      , callbacks_disc_ext=None
                      , callbacks_gen_disc=None
                      , validation_data_gen=None #?
                      , validation_steps=None
                      , validation_freq=1 #?
                      , class_weight=None #?
                      , max_queue_size=10
                      , workers=1
                      , use_multiprocessing=False #?
                      , shuffle=True
                      , initial_epoch=0): #?
        """Train the GAN model with the generator progressively.
        
        Parameters
        ----------
        generator: Generator
            Training data generator.
        verbose: Integer 
            Verbose mode (default=1).
        callback_disc_ext: list
            disc_ext callbacks (default=None).
        callback_gen_disc: list
            gen_disc callbacks (default=None).
        validation_data_gen: Generator or Sequence
            Validation generator or sequence (default=None).
        validation_steps: Integer
            Validation steps (default=None).
        validation_freq: Integer
            Validation frequency (default=1).
        max_queue_size: Integer
            Maximum size for the generator queue (default: 10).
        workers: Integer
            Maximum number of processes to get samples (default: 1, 0: main thread).
        use_multiprocessing: Boolean
            Multi-processing flag (default: False).
        shuffle: Boolean
            Batch shuffling flag (default: True).
        initial_epoch: Integer
            Initial epoch (default: 1).
        
        Returns
        -------
        Training history.
            tuple
        """
        
        # Check exception.
        do_validation = bool(validation_data_gen)
        if do_validation:
            assert hasattr(validation_data_gen, 'next') or \
                hasattr(validation_data_gen, '__next') or \
                isinstance(validation_data_gen, Sequence)
            
            if not isinstance(validation_data_gen, Sequence):
                assert validation_steps #?
        
            assert isinstance(validation_freq, int)
                    
        if not isinstance(generator, Sequence) and use_multiprocessing and workers > 1:
            warnings.warn(UserWarning('For multiprocessing, use the instance of Sequence.'))

        # Initialize the results directory
        if not os.path.isdir(os.path.join('results')):
            os.mkdir(os.path.join('results'))
        else:
            shutil.rmtree(os.path.join('results'))
            os.mkdir(os.path.join('results'))
            
        try:                    
            # Get the validation generator and output generator.
            if do_validation:
                if workers > 0:
                    if isinstance(validation_data_gen, Sequence):
                        val_enq = OrderedEnqueuer(validation_data_gen
                                                  , use_multiprocessing=use_multiprocessing) # shuffle?
                        validation_steps = validation_steps or len(validation_data_gen)
                    else:
                        val_enq = GeneratorEnqueuer(validation_data_gen
                                                    , use_multiprocessing=use_multiprocessing)
                    
                    val_enq.start(workers=workers, max_queue_size=max_queue_size)
                    val_generator = val_enq.get()
                else:
                    if isinstance(validation_data_gen, Sequence):
                        val_generator = iter_sequence_infinite(validation_data_gen)
                        validation_steps = validation_steps or len(validation_data_gen)
                    else:
                        val_generator = validation_data_gen 
            
            if workers > 0:
                if isinstance(generator, Sequence):
                    enq = OrderedEnqueuer(generator
                                      , use_multiprocessing=use_multiprocessing
                                      , shuffle=shuffle)
                else:
                    enq = GeneratorEnqueuer(generator
                                            , use_multiprocessing=use_multiprocessing)
                    
                enq.start(workers=workers, max_queue_size=max_queue_size)
                output_generator = enq.get()
            else:
                if isinstance(generator, Sequence):
                    output_generator = iter_sequence_infinite(generator)
                else:
                    output_generator = generator
                 
            # Callbacks.            
            # disc_ext.
            #out_labels_disc_ext = self.disc_ext.metrics_names if hasattr(self.disc_ext, 'metrics_names') else []
            out_labels_disc_ext = ['loss'] + [v.name for v in self.disc_ext.loss_functions]
            callback_metrics_disc_ext = out_labels_disc_ext \
                + ['val_' + out_label for out_label in out_labels_disc_ext]
            self.disc_ext.history = cbks.History()
            _callbacks = [cbks.BaseLogger(stateful_metrics=[])]
            if verbose:
                _callbacks.append(cbks.ProgbarLogger(count_mode='steps'
                                                     , stateful_metrics=[]))
                
            # Tensorboard callback.
            callback_tb = TensorBoard(log_dir='.\\logs'
                                           , histogram_freq=1
                                           , batch_size=self.hps['mini_batch_size']
                                           , write_graph=True
                                           , write_grads=True
                                           , write_images=True
                                           , update_freq='batch')
            _callbacks.append(callback_tb)
                            
            _callbacks += (callbacks_disc_ext or []) + [self.disc_ext.history]
            callbacks_disc_ext = cbks.configure_callbacks(_callbacks
                                                          , self.disc_ext
                                                          , do_validation=do_validation
                                                          , epochs=self.hps['epochs']
                                                          , steps_per_epoch=self.hps['batch_step'] * self.hps['disc_k_step']
                                                          , samples=self.hps['batch_step'] * self.hps['disc_k_step']
                                                          , verbose=1
                                                          , mode=ModeKeys.TRAIN)  
          
            # gen_disc.
            #out_labels_gen_disc = self.gen_disc.metrics_names if hasattr(self.gen_disc, 'metrics_names') else []
            out_labels_gen_disc = ['loss'] + [v.name for v in self.gen_disc.loss_functions]
            callback_metrics_gen_disc = out_labels_gen_disc \
                + ['val_' + out_label for out_label in out_labels_gen_disc]
            self.gen_disc.history = cbks.History()
            _callbacks = [cbks.BaseLogger(stateful_metrics=[])]
            if verbose:
                _callbacks.append(cbks.ProgbarLogger(count_mode='steps'
                                                     , stateful_metrics=[]))
            
            # Tensorboard callback.
            callback_tb = TensorBoard(log_dir='.\\logs'
                                           , histogram_freq=1
                                           , batch_size=self.hps['mini_batch_size']
                                           , write_graph=True
                                           , write_grads=True
                                           , write_images=True
                                           , update_freq='batch')
            _callbacks.append(callback_tb)
            
            _callbacks += (callbacks_gen_disc or []) + [self.gen_disc.history]
            callbacks_gen_disc = cbks.configure_callbacks(_callbacks
                                                          , self.gen_disc
                                                          , do_validation=do_validation
                                                          , epochs=self.hps['epochs']
                                                          , steps_per_epoch=self.hps['batch_step']
                                                          , samples=self.hps['batch_step']
                                                          , verbose=1
                                                          , mode=ModeKeys.TRAIN) 

            aggr_metrics_disc_ext = training_utils.MetricsAggregator(True, steps=self.hps['batch_step'])
            aggr_metrics_gen_disc = training_utils.MetricsAggregator(True, steps=self.hps['batch_step'])
            
            # Train.
            callbacks_disc_ext.model.stop_training = False
            callbacks_gen_disc.model.stop_training = False  
            callbacks_disc_ext._call_begin_hook(ModeKeys.TRAIN)
            callbacks_gen_disc._call_begin_hook(ModeKeys.TRAIN)
            
            initial_epoch = self.disc_ext._maybe_load_initial_epoch_from_ckpt(initial_epoch, ModeKeys.TRAIN) #?                                  
            
            for e_i in range(initial_epoch, self.hps['epochs']):
                if callbacks_disc_ext.model.stop_training or callbacks_gen_disc.model.stop_training:
                    break
                
                self.disc_ext.reset_metrics()
                self.gen_disc.reset_metrics()
                    
                epochs_log_disc_ext = {}
                epochs_log_gen_disc = {}
                                                   
                callbacks_disc_ext.on_epoch_begin(e_i, epochs_log_disc_ext)
                callbacks_gen_disc.on_epoch_begin(e_i, epochs_log_gen_disc)

                num_samples = None
                for s_i in range(self.hps['batch_step']):
                    for k_i in range(self.hps['disc_k_step']):
                        # Build batch logs.
                        k_batch_logs = {'batch': self.hps['disc_k_step'] * s_i + k_i + 1, 'size': self.hps['mini_batch_size']}
                        callbacks_disc_ext._call_batch_hook(ModeKeys.TRAIN
                                                            , 'begin'
                                                            , self.hps['disc_k_step'] * s_i + k_i + 1
                                                            , k_batch_logs)
                                                
                        inputs, outputs = gen_disc_ext_data_fun(output_generator)
                        outs = self.disc_ext.train_on_batch(inputs
                                 , outputs
                                 , class_weight=class_weight
                                 , reset_metrics=True) #?
                        del inputs, outputs
                        outs = to_list(outs) #?
                        
                        if s_i == 0:
                            aggr_metrics_disc_ext.create(outs)
                        
                        metrics_names = ['disc_ext_loss'] + [v.name for v in self.disc_ext.loss_functions]                             
                            
                        for l, o in zip(metrics_names, outs):
                            k_batch_logs[l] = o                        
                                    
                        callbacks_disc_ext._call_batch_hook(ModeKeys.TRAIN
                                                            , 'end'
                                                            , self.hps['disc_k_step'] * s_i + k_i + 1
                                                            , k_batch_logs)
                        print('\n', k_batch_logs)
                        
                    # Build batch logs.
                    batch_logs = {'batch': s_i + 1, 'size': self.hps['mini_batch_size']}
                    callbacks_gen_disc._call_batch_hook(ModeKeys.TRAIN
                                                        , 'begin'
                                                        , s_i
                                                        , batch_logs)

                    inputs, outputs = gen_gen_disc_data_fun(output_generator)
                    outs = self.gen_disc.train_on_batch(inputs
                                                        , outputs
                                                        , class_weight=class_weight
                                                        , reset_metrics=False)
                    del inputs, outputs
                    outs = to_list(outs)
                    
                    if s_i == 0:
                        aggr_metrics_gen_disc.create(outs)
                    
                    metrics_names = ['gen_disc_loss'] + [v.name for v in self.gen_disc.loss_functions]
                    
                    for l, o in zip(metrics_names, outs): #?
                        batch_logs[l] = o
        
                    callbacks_gen_disc._call_batch_hook(ModeKeys.TRAIN
                                                        , 'end'
                                                        , s_i
                                                        , batch_logs)
                    print('\n', batch_logs)
                
                aggr_metrics_disc_ext.finalize()
                aggr_metrics_gen_disc.finalize()
                
                # Make epochs log.
                outs_disc_ext = to_list(aggr_metrics_disc_ext.results)
                for out_label, out in zip(out_labels_disc_ext, outs_disc_ext):
                    epochs_log_disc_ext[out_label] = out
                                
                outs_gen_disc = to_list(aggr_metrics_gen_disc.results)
                for out_label, out in zip(out_labels_gen_disc, outs_gen_disc):
                    epochs_log_gen_disc[out_label] = out
                    
                # Do validation.
                if do_validation: #?
                    if e_i % validation_freq == 0: #?
                        # disc_ext.
                        val_outs_disc_ext = self._evaluate_disc_ext(val_generator
                                                                      , gen_disc_ext_data_fun
                                                                      , callbacks=callbacks_disc_ext
                                                                      , workers=0)
                        
                        # gen_disc.
                        val_outs_gen_disc = self._evaluate_gen_disc(val_generator
                                                                      , gen_gen_disc_data_fun
                                                                      , callbacks=callbacks_gen_disc
                                                                      , workers=0)                                            
                        # Make epochs log.
                        val_outs_disc_ext = to_list(val_outs_disc_ext)
                        for out_label, val_out in zip(out_labels_disc_ext, val_outs_disc_ext):
                            epochs_log_disc_ext['val_' + out_label] = val_out
                                        
                        val_outs_gen_disc = to_list(val_outs_gen_disc)
                        for out_label, val_out in zip(out_labels_gen_disc, val_outs_gen_disc):
                            epochs_log_gen_disc['val_' + out_label] = val_out
                                                                
                callbacks_disc_ext.on_epoch_end(e_i, epochs_log_disc_ext)
                callbacks_gen_disc.on_epoch_end(e_i, epochs_log_gen_disc)
                                
            self.disc_ext._successful_loop_finish = True
            self.gen_disc._successful_loop_finish = True
                
            callbacks_disc_ext._call_end_hook(ModeKeys.TRAIN)
            callbacks_gen_disc._call_end_hook(ModeKeys.TRAIN) 
        finally:
            try:
                if enq is not None:
                    enq.stop()
            finally:
                if val_enq is not None:
                    val_enq.stop()

        return self.disc_ext.history, self.gen_disc.history

    def save_gan_model(self):
        """Save the GAN model."""
        assert hasattr(self, 'disc_ext_init') and hasattr(self, 'gen_disc_init')
        
        with CustomObjectScope(self.custom_objects):
            self.disc_ext_init.save(self.DISC_EXT_PATH)
            self.gen_disc_init.save(self.GEN_DISC_PATH)

    def _evaluate_disc_ext(self
                      , generator
                      , gen_disc_ext_data_func
                      , verbose=1
                      , callbacks=None
                      , max_queue_size=10
                      , workers=1
                      , use_multiprocessing=False):
        """Evaluate the extended discriminator.
        
        Parameters
        ----------
        generator: Generator
            Test data generator.
        verbose: Integer 
            Verbose mode (default=1).
        callback: list
            Callbacks (default=None).
        max_queue_size: Integer
            Maximum size for the generator queue (default: 10).
        class_weight: TODO. 
            TODO.
        workers: Integer
            Maximum number of processes to get samples (default: 1, 0: main thread).
        use_multiprocessing: Boolean
            Multi-processing flag (default: False).
.        
        Returns
        -------
        Training history.
            tuple.
        """

        # Check exception.                    
        if not isinstance(generator, Sequence) and use_multiprocessing and workers > 1:
            warnings.warn(UserWarning('For multiprocessing, use the instance of Sequence.'))
        
        out_labels = ['loss'] + [v.name for v in self.disc_ext.loss_functions] #?                                                   
        aggr_metrics = training_utils.MetricsAggregator(True, steps=self.hps['batch_step'])
        
        # Evaluate.
        callbacks._call_begin_hook(ModeKeys.TEST)
        self.disc_ext.reset_metrics()
        epochs_log= {}      
        callbacks.on_epoch_begin(0, epochs_log)

        for k_i in range(self.hps['batch_step']):
            # Build batch logs.
            k_batch_logs = {'batch': self.hps['batch_step'] * k_i + 1, 'size': self.hps['mini_batch_size']}
            callbacks._call_batch_hook(ModeKeys.TEST
                                                , 'begin'
                                                , self.hps['batch_step'] * k_i + 1
                                                , k_batch_logs)
            
            inputs, outputs = gen_disc_ext_data_func(generator)
            outs = self.disc_ext.test_on_batch(inputs, outputs, reset_metrics=False) #?  
            del inputs, outputs
            outs = to_list(outs) #?
            
            if k_i == 0:
                aggr_metrics.create(outs)
            
            metrics_names = ['loss'] + [v.name for v in self.disc_ext.loss_functions]                            
                
            for l, o in zip(metrics_names, outs):
                k_batch_logs[l] = o                        

            ws = self.gen.get_weights()
            res = []
            for w in ws:
                res.append(np.isfinite(w).all())
            res = np.asarray(res)
            
            callbacks._call_batch_hook(ModeKeys.TEST
                                                , 'end'
                                                , self.hps['batch_step'] * k_i + 1
                                                , k_batch_logs)
            print('\n', k_batch_logs)
                            
        aggr_metrics.finalize()
        
        # Make epochs log.
        outs = to_list(aggr_metrics.results)
        for out_label, out in zip(out_labels, outs):
            epochs_log[out_label] = out
                                                                      
        callbacks.on_epoch_end(0, epochs_log)
                    
        self.disc_ext._successful_loop_finish = True
            
        callbacks._call_end_hook(ModeKeys.TEST)

        return aggr_metrics.results
    
    def _evaluate_gen_disc(self
                      , generator
                      , gen_gen_disc_data_func
                      , verbose=1
                      , callbacks=None
                      , max_queue_size=10
                      , workers=1
                      , use_multiprocessing=False):
        """Evaluate the generator via discriminator. #?
        
        Parameters
        ----------
        generator: Generator
            Test data generator.
        verbose: Integer 
            Verbose mode (default=1).
        callback: list
            Callbacks (default=None).
        max_queue_size: Integer
            Maximum size for the generator queue (default: 10).
        class_weight: TODO. 
            TODO.
        workers: Integer
            Maximum number of processes to get samples (default: 1, 0: main thread).
        use_multiprocessing: Boolean
            Multi-processing flag (default: False).
.        
        Returns
        -------
        Training history.
            tuple.
        """

        # Check exception.                    
        if not isinstance(generator, Sequence) and use_multiprocessing and workers > 1:
            warnings.warn(UserWarning('For multiprocessing, use the instance of Sequence.'))
        
        out_labels = ['loss'] + [v.name for v in self.gen_disc.loss_functions] #?                                                   
        aggr_metrics = training_utils.MetricsAggregator(True, steps=self.hps['batch_step'])
        
        # Evaluate.
        callbacks._call_begin_hook(ModeKeys.TEST)
        self.gen_disc.reset_metrics()
        epochs_log= {}      
        callbacks.on_epoch_begin(0, epochs_log)

        for s_i in range(self.hps['batch_step']):                
            # Build batch logs.
            batch_logs = {'batch': s_i + 1, 'size': self.hps['mini_batch_size']}
            callbacks._call_batch_hook(ModeKeys.TEST
                                                , 'begin'
                                                , s_i
                                                , batch_logs)

            inputs, outputs = gen_gen_disc_data_func(generator)
            outs = self.gen_disc.test_on_batch(inputs, outputs, reset_metrics=False) #?  
            del inputs, outputs
            outs = to_list(outs)
            
            if s_i == 0:
                aggr_metrics.create(outs)
            
            metrics_names = ['loss'] + [v.name for v in self.gen_disc.loss_functions]
            
            for l, o in zip(metrics_names, outs): #?
                batch_logs[l] = o

            callbacks._call_batch_hook(ModeKeys.TEST
                                                , 'end'
                                                , s_i
                                                , batch_logs)
            print('\n', batch_logs)
        
        aggr_metrics.finalize()
        
        # Make epochs log.
        outs = to_list(aggr_metrics.results)
        for out_label, out in zip(out_labels, outs):
            epochs_log[out_label] = out

        callbacks.on_epoch_end(0, epochs_log)
                           
        self.gen_disc._successful_loop_finish = True
            
        callbacks._call_end_hook(ModeKeys.TEST)

        return aggr_metrics.results
        
    def generate(self, inputs, *args, **kwargs):
        """Generate.
        
        Parameters
        ----------
        inputs: Numpy array, list or tuple.
            Inputs.
        """
        inputs = inputs if isinstance(inputs, [list, tuple]) else [inputs]
        
        if self.conf['multi_gpu']:
            if hasattr(self, 'gen_p'):
                results = self.gen_p.predict(inputs)
            else:
                self.gen_p = multi_gpu_model()
                results = self.gen_p.predict(inputs)
        else:
            results = self.gen.predict(inputs)
        
        return results
    
def compose_gan_with_mode(gan_model, mode):
    """Compose the GAN model with mode.
    
    Parameters
    ----------
    gan_model: AbstractGAN instance.
        GAN model.
    mode: Integer.
        GAN composing mode.
    """
    
    # Check exception.
    assert hasattr(gan_model, 'gen') and hasattr(gan_model, 'disc')
    
    if mode == STYLE_GAN:
        # Compose gan.                    
        # Compose disc_ext.
        # disc.
        x_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in gan_model.disc.inputs]  
        x_outputs = [gan_model.disc(x_inputs)] if len(gan_model.disc.outputs) == 1 else gan_model.disc(x_inputs) #? 
        
        # gen and disc.
        z_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in gan_model.gen.inputs] 
               
        gan_model.gen.trainable = False
        for layer in gan_model.gen.layers: layer.trainable = False
        z_outputs = [gan_model.gen(z_inputs)] if len(gan_model.gen.outputs) == 1 else gan_model.gen(z_inputs)
        
        gan_model.disc.trainable = True
        for layer in gan_model.disc.layers: layer.trainable = True
        x2_outputs = [gan_model.disc(z_outputs)] if len(gan_model.disc.outputs) == 1 else gan_model.disc(z_outputs)
        
        gan_model.disc_ext_init = ModelExt(inputs=x_inputs + z_inputs
                                           , outputs=x_outputs + x2_outputs
                                           , name='disc_ext')
        if gan_model.conf['multi_gpu']:
            gan_model.disc_ext = multi_gpu_model(gan_model.disc_ext_init, gpus=gan_model.conf['num_gpus']) # Name?   
        else:
            gan_model.disc_ext = gan_model.disc_ext_init
                
        # Compose gen_disc.
        z_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in gan_model.gen_inputs] 
        
        gan_model.gen.trainable = True
        for layer in gan_model.gen.layers: layer.trainable = True    
        z_outputs = [gan_model.gen(z_inputs)] if len(gan_model.gen.outputs) == 1 else gan_model.gen(z_inputs)
        
        gan_model.disc.trainable = False
        for layer in gan_model.disc.layers: layer.trainable = False
        z_p_outputs = [gan_model.disc(z_outputs)] if len(gan_model.disc.outputs) == 1 else gan_model.disc(z_outputs)

        gan_model.gen_disc_init = ModelExt(inputs=z_inputs
                                           , outputs=z_p_outputs
                                           , name='gen_disc')
        if gan_model.conf['multi_gpu']:
            gan_model.gen_disc = multi_gpu_model(gan_model.gen_disc_init, gpus=gan_model.conf['num_gpus'])
        else:
            gan_model.gen_disc = gan_model.gen_disc_init 
    elif mode == PIX2PIX_GAN:
        # Compose gan.                    
        # Compose disc_ext.
        # disc.
        x_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in gan_model.disc.inputs]  
        x_outputs = [gan_model.disc(x_inputs)] if len(gan_model.disc.outputs) == 1 else gan_model.disc(x_inputs) #? 
        
        # gen and disc.
        z_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in gan_model.gen.inputs]
               
        gan_model.gen.trainable = False
        for layer in gan_model.gen.layers: layer.trainable = False
        z_outputs = [gan_model.gen(z_inputs)] if len(gan_model.gen.outputs) == 1 else gan_model.gen(z_inputs)
        
        gan_model.disc.trainable = True
        for layer in gan_model.disc.layers: layer.trainable = True
        
        # Get image inputs.
        image_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in gan_model.disc.inputs \
                         if 'image' in t.name]        
                        
        x2_outputs = [gan_model.disc(image_inputs + z_outputs)] \
            if len(gan_model.disc.outputs) == 1 else gan_model.disc(image_inputs + z_outputs)
        
        gan_model.disc_ext_init = ModelExt(inputs=x_inputs + z_inputs + image_inputs
                                           , outputs=x_outputs + x2_outputs
                                           , name='disc_ext')
        if gan_model.conf['multi_gpu']:
            gan_model.disc_ext = multi_gpu_model(gan_model.disc_ext_init, gpus=gan_model.conf['num_gpus'])
        else:
            gan_model.disc_ext = gan_model.disc_ext_init    
                
        # Compose gen_disc.
        z_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in gan_model.gen.inputs] 
        
        gan_model.gen.trainable = True
        for layer in gan_model.gen.layers: layer.trainable = True    
        z_outputs = [gan_model.gen(z_inputs)] if len(gan_model.gen.outputs) == 1 else gan_model.gen(z_inputs)
        
        gan_model.disc.trainable = False
        for layer in gan_model.disc.layers: layer.trainable = False

        # Get image inputs.
        image_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in gan_model.disc.inputs \
                        if 'image' in t.name]
         
        z_p_outputs = [gan_model.disc(image_inputs + z_outputs)] \
            if len(gan_model.disc.outputs) == 1 else gan_model.disc(image_inputs + z_outputs)

        gan_model.gen_disc_init = ModelExt(inputs=z_inputs + image_inputs
                                           , outputs=z_p_outputs + z_outputs 
                                           , name='gen_disc')
        if gan_model.conf['multi_gpu']:
            gan_model.gen_disc = multi_gpu_model(gan_model.gen_disc_init, gpus=gan_model.conf['num_gpus'])
        else:
            gan_model.gen_disc = gan_model.gen_disc_init
    else:
        ValueError('mode is not valid.')            