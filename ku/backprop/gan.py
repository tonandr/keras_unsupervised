from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC, abstractmethod
import os
import warnings
import shutil

import numpy as np

import tensorflow as tf
import tensorflow_core.python.keras.backend as K 
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.utils import Sequence, GeneratorEnqueuer, OrderedEnqueuer
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Lambda
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError

from tensorflow_core.python.keras.utils.generic_utils import to_list, CustomObjectScope
from tensorflow_core.python.keras.utils.data_utils import iter_sequence_infinite
from tensorflow_core.python.keras import callbacks as cbks
from tensorflow_core.python.keras.engine import training_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys #?

from ..engine_ext import ModelExt
from ..loss_ext import WGANLoss, WGANGPLoss
from ..loss_ext import SoftPlusInverseLoss, SoftPlusLoss, RPenaltyLoss

# GAN mode.
STYLE_GAN_REGULAR = 0
STYLE_GAN_WGAN_GP = 1
STYLE_GAN_SOFTPLUS_INVERSE_R1_GP = 2
LSGAN = 3
PIX2PIX_GAN = 4
CYCLE_GAN = 5

# Loss configuration type.
LOSS_CONF_TYPE_NON_SATURATION_REGULAR = 0
LOSS_CONF_TYPE_WGAN_GP = 1
LOSS_CONF_TYPE_NON_SATURATION_SOFTPLUS_R1_GP = 2
LOSS_CONF_TYPE_LS = 3

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
    if lc_type == LOSS_CONF_TYPE_NON_SATURATION_REGULAR:
        loss_conf = {'disc_ext_losses': [BinaryCrossentropy(from_logits=True), BinaryCrossentropy(from_logits=True)]
                    , 'disc_ext_loss_weights': [1.0, 1.0]
                    , 'gen_disc_losses': [BinaryCrossentropy(from_logits=True)]
                    , 'gen_disc_loss_weights': [1.0]}
    elif lc_type == LOSS_CONF_TYPE_WGAN_GP:
        loss_conf = {'disc_ext_losses': [WGANLoss()
                                , WGANLoss()
                                , WGANGPLoss(model=kwargs['model']
                                    , input_variable_orders=kwargs['input_variable_orders']
                                    , wgan_lambda=hps['wgan_lambda']
                                    , wgan_target=hps['wgan_target'])]
                    , 'disc_ext_loss_weights': [-1.0, 1.0, 1.0]
                    , 'gen_disc_losses': [WGANLoss()]
                    , 'gen_disc_loss_weights': [-1.0]}
    elif lc_type == LOSS_CONF_TYPE_NON_SATURATION_SOFTPLUS_R1_GP :
        loss_conf = {'disc_ext_losses': [SoftPlusInverseLoss()
                                , RPenaltyLoss(model=kwargs['model']
                                               , input_variable_orders=kwargs['input_variable_orders'] 
                                               , r_gamma=hps['r_gamma'])
                                , SoftPlusLoss()]
                    , 'disc_ext_loss_weights': [1.0, 1.0, 1.0]
                    , 'gen_disc_losses': [SoftPlusInverseLoss()]
                    , 'gen_disc_loss_weights': [1.0]}
    elif lc_type == LOSS_CONF_TYPE_LS:
        loss_conf = {'disc_ext_losses': [MeanSquaredError(), MeanSquaredError()]
                    , 'disc_ext_loss_weights': [1.0, 1.0]
                    , 'gen_disc_losses': [MeanSquaredError()]
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
                                                          
            self.custom_objects['ModelExt'] = ModelExt                                                          
            with CustomObjectScope(self.custom_objects):
                # disc_ext.
                '''
                self.disc_ext = load_model(self.DISC_EXT_PATH
                                           , custom_objects=self.custom_objects
                                           , compile=False) #?
                '''
                 
                # gen_disc.
                self.gen_disc = load_model(self.GEN_DISC_PATH
                                           , custom_objects=self.custom_objects
                                           , compile=False) #?
                                        
                # gen, disc.
                self.gen = self.gen_disc.get_layer('gen')
                self.disc = self.gen_disc.get_layer('disc')
                
                if conf['multi_gpu']: #?
                    """
                    self.disc_ext = multi_gpu_model(self.disc_ext
                                                    , gpus=self.conf['num_gpus']
                                                    , name='disc_ext')
                    
                    self.disc_ext.compile(optimizer=self.disc_ext.optimizer
                                          , loss=self.disc_ext.losses
                                          , loss_weights=self.disc_ext.loss_weights
                                          , run_eagerly=True)
                    """
                    
                    self.gen_disc = multi_gpu_model(self.gen_disc
                                                    , gpus=self.conf['num_gpus']
                                                    , name='gen_disc')
                    """
                    self.gen_disc.compile(optimizer=self.gen_disc.optimizer
                                          , loss=self.gen_disc.losses
                                          , loss_weights=self.gen_disc.loss_weights
                                          , run_eagerly=True)
                    """
                    
                    self.gen = multi_gpu_model(self.gen, gpus=self.conf['num_gpus'], name='gen') #?
                    self.disc = multi_gpu_model(self.disc, gpus=self.conf['num_gpus'], name='disc') #?

            
            
            #self._is_gan_compiled = True        
    
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
    
    def compose_gan_with_mode(self, mode):
        """Compose gan with mode.
        
        Parameters
        ----------
        mode: Integer.
            GAN model composing mode.
        """
        disc_ext, gen_disc = compose_gan_with_mode(self.gen, self.disc, mode)
        self.disc_ext = disc_ext
        self.gen_disc = gen_disc
        
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

        self.gen.trainable = False
        for layer in self.gen.layers: layer.trainable = False
        
        self.disc.trainable = True
        for layer in self.disc.layers: layer.trainable = True

        self.disc_ext.compile(optimizer=disc_ext_opt
                         , loss=disc_ext_losses
                         , loss_weights=disc_ext_loss_weights
                         , metrics=disc_ext_metrics
                         , run_eagerly=True)
        
        self.gen.trainable = True
        for layer in self.gen.layers: layer.trainable = True
        
        self.disc.trainable = False
        for layer in self.disc.layers: layer.trainable = False        
        
        self.gen_disc.compile(optimizer=gen_disc_opt
                         , loss=gen_disc_losses
                         , loss_weights=gen_disc_loss_weights
                         , metrics=gen_disc_metrics
                         , run_eagerly=True)
        self._is_gan_compiled = True
    
    @abstractmethod
    def gen_disc_ext_data_fun(self, generator, gen_prog_depth=None, disc_prog_depth=None, *args, **kwargs):
        """Generate disc_ext data.
        
        Parameters
        ----------
        generator: Generator.
            Data generator.
        gen_prog_depth: Integer.
            Partial generator model's layer depth (default: None).
        disc_prog_depth: Integer.
            Partial discriminator model's layer depth (default: None).
        """
        raise NotImplementedError('gen_disc_ext_data_fun is not implemented.')

    @abstractmethod
    def gen_gen_disc_data_fun(self, generator, gen_prog_depth=None, disc_prog_depth=None, *args, **kwargs):
        """Generate disc_ext data.
        
        Parameters
        ----------
        generator: Generator.
            Data generator.
        gen_prog_depth: Integer.
            Partial generator model's layer depth (default: None).
        disc_prog_depth: Integer.
            Partial discriminator model's layer depth (default: None).
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
        generator: Generator.
            Training data generator.
        gen_disc_ext_data_fun: Function.
            Data generating function for disc_ext.
        gen_gen_disc_data_fun: Function.
            Data generating function for gen_disc.
        verbose: Integer. 
            Verbose mode (default=1).
        callback_disc_ext: list.
            disc_ext callbacks (default=None).
        callback_gen_disc: list.
            gen_disc callbacks (default=None).
        validation_data_gen: Generator or Sequence.
            Validation generator or sequence (default=None).
        validation_steps: Integer.
            Validation steps (default=None).
        validation_freq: Integer.
            Validation frequency (default=1).
        class_weight: Numpy array. ?
            Class weight (default=None).
        max_queue_size: Integer.
            Maximum size for the generator queue (default: 10).
        workers: Integer.
            Maximum number of processes to get samples (default: 1, 0: main thread).
        use_multiprocessing: Boolean.
            Multi-processing flag (default: False).
        shuffle: Boolean.
            Batch shuffling flag (default: True).
        initial_epoch: Integer.
            Initial epoch (default: 0).
        save_f: Boolean.
            Model saving flag (default: True).
        
        Returns
        -------
        Training history.
            Tuple.
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
        
        enq = None
        val_enq = None    
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
            out_labels_disc_ext = self.disc_ext.metrics_names if hasattr(self.disc_ext, 'metrics_names') else []
            out_labels_disc_ext = ['loss'] + [v.name for v in self.disc_ext.loss_functions]
            #self.disc_ext.history = cbks.History()
            _callbacks = [] #cbks.BaseLogger(stateful_metrics=[])]
            
            '''
            if verbose:
                _callbacks.append(cbks.ProgbarLogger(count_mode='steps'
                                                     , stateful_metrics=[]))
            '''
                
            # Tensorboard callback.
            callback_tb = TensorBoard(log_dir='logs'
                                           , histogram_freq=1
                                           , write_graph=True
                                           , write_images=True
                                           , update_freq='batch')
            #_callbacks.append(callback_tb) #?
                            
            #_callbacks += (callbacks_disc_ext or []) + [self.disc_ext.history]
            callbacks_disc_ext = cbks.configure_callbacks(_callbacks
                                                          , self.disc_ext
                                                          , do_validation=do_validation
                                                          , epochs=self.hps['epochs']
                                                          , steps_per_epoch=self.hps['batch_step'] * self.hps['disc_k_step']
                                                          , samples=self.hps['batch_step'] * self.hps['disc_k_step']
                                                          , verbose=0
                                                          , mode=ModeKeys.TRAIN)
            progbar_disc_ext = training_utils.get_progbar(self.disc_ext, 'steps')
            progbar_disc_ext.params = callbacks_disc_ext.params
            progbar_disc_ext.params['verbose'] = verbose  
          
            # gen_disc.
            out_labels_gen_disc = self.gen_disc.metrics_names if hasattr(self.gen_disc, 'metrics_names') else []
            out_labels_gen_disc = ['loss'] + [v.name for v in self.gen_disc.loss_functions]
            #self.gen_disc.history = cbks.History()
            _callbacks = [] #cbks.BaseLogger(stateful_metrics=[])]
            
            '''
            if verbose:
                _callbacks.append(cbks.ProgbarLogger(count_mode='steps'
                                                     , stateful_metrics=[]))
            '''
            
            # Tensorboard callback.
            callback_tb = TensorBoard(log_dir='logs'
                                           , histogram_freq=1
                                           , write_graph=True
                                           , write_images=True
                                           , update_freq='batch')
            #_callbacks.append(callback_tb)
            
            #_callbacks += (callbacks_gen_disc or []) + [self.gen_disc.history]
            callbacks_gen_disc = cbks.configure_callbacks(_callbacks
                                                          , self.gen_disc
                                                          , do_validation=do_validation
                                                          , epochs=self.hps['epochs']
                                                          , steps_per_epoch=self.hps['batch_step']
                                                          , samples=self.hps['batch_step']
                                                          , verbose=0
                                                          , mode=ModeKeys.TRAIN)
            progbar_gen_disc = training_utils.get_progbar(self.gen_disc, 'steps')
            progbar_gen_disc.params = callbacks_gen_disc.params
            progbar_gen_disc.params['verbose'] = verbose  
            
            # Train.
            callbacks_disc_ext.model.stop_training = False
            callbacks_gen_disc.model.stop_training = False  
            callbacks_disc_ext._call_begin_hook(ModeKeys.TRAIN)
            callbacks_gen_disc._call_begin_hook(ModeKeys.TRAIN)
            progbar_disc_ext.on_train_begin()
            progbar_gen_disc.on_train_begin()
                        
            initial_epoch = self.disc_ext._maybe_load_initial_epoch_from_ckpt(initial_epoch, ModeKeys.TRAIN) #?                                  
            
            pre_e_i = -1
            for e_i in range(initial_epoch, self.hps['epochs']):
                aggr_outputs_disc_ext = training_utils.MetricsAggregator(True, steps=(self.hps['batch_step'] * self.hps['disc_k_step']))
                aggr_outputs_gen_disc = training_utils.MetricsAggregator(True, steps=self.hps['batch_step'])
                                
                if callbacks_disc_ext.model.stop_training or callbacks_gen_disc.model.stop_training:
                    break
                
                self.disc_ext.reset_metrics() #?
                self.gen_disc.reset_metrics() 
                    
                epochs_log_disc_ext = {}
                epochs_log_gen_disc = {}
                                                   
                callbacks_disc_ext.on_epoch_begin(e_i, epochs_log_disc_ext)
                callbacks_gen_disc.on_epoch_begin(e_i, epochs_log_gen_disc)
                progbar_disc_ext.on_epoch_begin(e_i, epochs_log_disc_ext)
                progbar_gen_disc.on_epoch_begin(e_i, epochs_log_gen_disc)

                for s_i in range(self.hps['batch_step']):
                    for k_i in range(self.hps['disc_k_step']):
                        # Build batch logs.
                        k_batch_logs = {'batch': self.hps['disc_k_step'] * s_i + k_i + 1, 'size': self.hps['batch_size']}
                        callbacks_disc_ext._call_batch_hook(ModeKeys.TRAIN
                                                            , 'begin'
                                                            , self.hps['disc_k_step'] * s_i + k_i + 1
                                                            , k_batch_logs)
                        progbar_disc_ext.on_batch_begin(self.hps['disc_k_step'] * s_i + k_i + 1, k_batch_logs)
                                                
                        inputs, outputs = gen_disc_ext_data_fun(output_generator)
                        
                        self.gen.trainable = False
                        for layer in self.gen.layers: layer.trainable = False
                        
                        self.disc.trainable = True
                        for layer in self.disc.layers: layer.trainable = True
                        
                        outs = self.disc_ext.train_on_batch(inputs
                                 , outputs
                                 , class_weight=class_weight
                                 , reset_metrics=True) #?
                        del inputs, outputs
                        outs = to_list(outs) #?
                        
                        outs_p = [np.asarray([v]) for v in outs]
                        
                        if pre_e_i != e_i and s_i == 0 and k_i == 0:
                            aggr_outputs_disc_ext.create(outs_p)
                        else:
                            aggr_outputs_disc_ext.aggregate(outs_p)
                        
                        metrics_names = ['disc_ext_loss'] + [v.name for v in self.disc_ext.loss_functions]                             
                            
                        #for l, o in zip(metrics_names, outs):
                        #    k_batch_logs[l] = o                        
                        k_batch_logs = cbks.make_logs(self.disc_ext, k_batch_logs, outs_p, ModeKeys.TRAIN)
                        
                        callbacks_disc_ext._call_batch_hook(ModeKeys.TRAIN
                                                            , 'end'
                                                            , self.hps['disc_k_step'] * s_i + k_i + 1
                                                            , k_batch_logs)
                        
                        progbar_disc_ext.on_batch_end( self.hps['disc_k_step'] * s_i + k_i + 1, k_batch_logs)
                        print('\n', k_batch_logs)
                        
                    # Build batch logs.
                    batch_logs = {'batch': s_i + 1, 'size': self.hps['batch_size']}
                    callbacks_gen_disc._call_batch_hook(ModeKeys.TRAIN
                                                        , 'begin'
                                                        , s_i
                                                        , batch_logs)
                    progbar_gen_disc.on_batch_begin(s_i, batch_logs)

                    inputs, outputs = gen_gen_disc_data_fun(output_generator)
                    
                    self.gen.trainable = True
                    for layer in self.gen.layers: layer.trainable = True
                    
                    self.disc.trainable = False
                    for layer in self.disc.layers: layer.trainable = False
                    
                    outs = self.gen_disc.train_on_batch(inputs
                                                        , outputs
                                                        , class_weight=class_weight
                                                        , reset_metrics=False)
                    del inputs, outputs
                    outs = to_list(outs)
                    outs_p = [np.asarray([v]) for v in outs]
                    
                    if pre_e_i != e_i and s_i == 0:
                        aggr_outputs_gen_disc.create(outs_p)
                    else:
                        aggr_outputs_gen_disc.aggregate(outs_p)
                    
                    metrics_names = ['gen_disc_loss'] + [v.name for v in self.gen_disc.loss_functions]
                    
                    #for l, o in zip(metrics_names, outs): #?
                    #    batch_logs[l] = o
                    batch_logs = cbks.make_logs(self.gen_disc, batch_logs, outs_p, ModeKeys.TRAIN)
        
                    callbacks_gen_disc._call_batch_hook(ModeKeys.TRAIN
                                                        , 'end'
                                                        , s_i
                                                        , batch_logs)
                    
                    progbar_gen_disc.on_batch_end(s_i, batch_logs)
                    print('\n', batch_logs)
                
                aggr_outputs_disc_ext.finalize()
                aggr_outputs_gen_disc.finalize()
                
                # Make epochs log.
                outs_disc_ext = to_list(aggr_outputs_disc_ext.results)
                #for out_label, out in zip(out_labels_disc_ext, outs_disc_ext):
                #    epochs_log_disc_ext[out_label] = out
                epochs_log_disc_ext = cbks.make_logs(self.disc_ext
                                                     , epochs_log_disc_ext
                                                     , outs_disc_ext
                                                     , ModeKeys.TRAIN)
                
                outs_gen_disc = to_list(aggr_outputs_gen_disc.results)
                #for out_label, out in zip(out_labels_gen_disc, outs_gen_disc):
                #    epochs_log_gen_disc[out_label] = out
                epochs_log_gen_disc = cbks.make_logs(self.gen_disc
                                                     , epochs_log_gen_disc
                                                     , outs_gen_disc 
                                                     , ModeKeys.TRAIN)
                    
                # Do validation.
                if do_validation: #?
                    if e_i % validation_freq == 0: #?
                        # disc_ext.
                        val_outs_disc_ext = self._evaluate_disc_ext(self.disc_ext
                                                                      , val_generator #?
                                                                      , gen_disc_ext_data_fun
                                                                      , callbacks=None #callbacks_disc_ext
                                                                      , workers=0
                                                                      , verbose=0)
                        
                        # gen_disc.
                        val_outs_gen_disc = self._evaluate_gen_disc(self.gen_disc
                                                                      , val_generator
                                                                      , gen_gen_disc_data_fun
                                                                      , callbacks=None #callbacks_gen_disc
                                                                      , workers=0
                                                                      , verbose=0)                                            
                        # Make epochs log.
                        val_outs_disc_ext = to_list(val_outs_disc_ext)
                        #for out_label, val_out in zip(out_labels_disc_ext, val_outs_disc_ext):
                        #    epochs_log_disc_ext['val_' + out_label] = val_out
                        epochs_log_disc_ext = cbks.make_logs(self.disc_ext
                                                     , epochs_log_disc_ext
                                                     , val_outs_disc_ext
                                                     , ModeKeys.TRAIN
                                                     , prefix='val_')
                                        
                        val_outs_gen_disc = to_list(val_outs_gen_disc)
                        #for out_label, val_out in zip(out_labels_gen_disc, val_outs_gen_disc):
                        #    epochs_log_gen_disc['val_' + out_label] = val_out
                        epochs_log_gen_disc = cbks.make_logs(self.gen_disc
                                                     , epochs_log_gen_disc
                                                     , val_outs_gen_disc
                                                     , ModeKeys.TRAIN
                                                     , prefix='val_')
                                                                
                callbacks_disc_ext.on_epoch_end(e_i, epochs_log_disc_ext)
                callbacks_gen_disc.on_epoch_end(e_i, epochs_log_gen_disc)
                progbar_disc_ext.on_epoch_end(e_i, epochs_log_disc_ext)
                progbar_gen_disc.on_epoch_end(e_i, epochs_log_gen_disc)
                
                if save_f:
                    self.save_gan_model()
                                
                pre_e_i = e_i
                                
            self.disc_ext._successful_loop_finish = True
            self.gen_disc._successful_loop_finish = True
                
            callbacks_disc_ext._call_end_hook(ModeKeys.TRAIN) # progress bar?
            callbacks_gen_disc._call_end_hook(ModeKeys.TRAIN) #
        finally:
            try:
                if enq:
                    enq.stop()
            finally:
                if val_enq:
                    val_enq.stop()

        return self.disc_ext.history, self.gen_disc.history

    def fit_generator_progressively(self
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
        
        enq = None
        val_enq = None     
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
                                           , write_graph=True
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
                                           , write_graph=True
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

                # Train disc_ext, gen_disc models progressively according to the schedule for epochs.
                # Make partial disc_ext, gen_disc.
                partial_gen = self.gen.create_prog_model(ModelExt.PROGRESSIVE_MODE_FORWARD
                                                         , self.nn_arch['gen_prog_depths'][e_i]
                                                         , self.nn_arch['gen_prog_fixed_layer_names'])
                partial_disc = self.disc.create_prog_model(ModelExt.PROGRESSIVE_MODE_BACKWARD
                                                         , self.nn_arch['disc_prog_depths'][e_i]
                                                         , self.nn_arch['disc_prog_fixed_layer_names'])
                partial_disc_ext, partial_gen_disc = compose_gan_with_mode(partial_gen
                                                                           , partial_disc
                                                                           , self.nn_arch['composing_mode']
                                                                           , multi_gpu=self.conf['multi_gpu']
                                                                           , num_gpus=self.conf['num_gpus'])
                
                for s_i in range(self.hps['batch_step']):
                    for k_i in range(self.hps['disc_k_step']):
                        # Build batch logs.
                        k_batch_logs = {'batch': self.hps['disc_k_step'] * s_i + k_i + 1, 'size': self.hps['batch_size']}
                        callbacks_disc_ext._call_batch_hook(ModeKeys.TRAIN
                                                            , 'begin'
                                                            , self.hps['disc_k_step'] * s_i + k_i + 1
                                                            , k_batch_logs)
                                                
                        inputs, outputs = gen_disc_ext_data_fun(output_generator
                                                                , gen_prog_depth=self.nn_arch['gen_prog_depths'][e_i]
                                                                , disc_prog_depth=self.nn_arch['disc_prog_depths'][e_i])
                        outs = partial_disc_ext.train_on_batch(inputs
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
                    batch_logs = {'batch': s_i + 1, 'size': self.hps['batch_size']}
                    callbacks_gen_disc._call_batch_hook(ModeKeys.TRAIN
                                                        , 'begin'
                                                        , s_i
                                                        , batch_logs)

                    inputs, outputs = gen_disc_ext_data_fun(output_generator
                                                                , gen_prog_depth=self.nn_arch['gen_prog_depths'][e_i]
                                                                , disc_prog_depth=self.nn_arch['disc_prog_depths'][e_i])
                    outs = partial_gen_disc.train_on_batch(inputs
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
                        val_outs_disc_ext = self._evaluate_disc_ext(partial_disc_ext
                                                                      , val_generator
                                                                      , gen_disc_ext_data_fun
                                                                      , callbacks=callbacks_disc_ext
                                                                      , workers=0)
                        
                        # gen_disc.
                        val_outs_gen_disc = self._evaluate_gen_disc(partial_gen_disc
                                                                      , val_generator
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
                if enq:
                    enq.stop()
            finally:
                if val_enq:
                    val_enq.stop()

        return self.disc_ext.history, self.gen_disc.history

    def save_gan_model(self):
        """Save the GAN model."""
        assert hasattr(self, 'disc_ext') and hasattr(self, 'gen_disc')
        
        with CustomObjectScope(self.custom_objects):
            self.disc_ext.save(self.DISC_EXT_PATH, save_format='h5')
            self.gen_disc.save(self.GEN_DISC_PATH, save_format='h5')

    def _evaluate_disc_ext(self
                      , disc_ext
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
        disc_ext: ModelExt.
            Discriminator extension.
        generator: Generator
            Test data generator.
        verbose: Integer 
            Verbose mode (default=1).
        callback: list
            Callbacks (default=None).
        max_queue_size: Integer
            Maximum size for the generator queue (default: 10).
        class_weight: Numpy array. ?
            Class weight.
        workers: Integer
            Maximum number of processes to get samples (default: 1, 0: main thread).
        use_multiprocessing: Boolean
            Multi-processing flag (default: False).
.        
        Returns
        -------
        Training history.
            Tuple.
        """

        # Check exception.                    
        if not isinstance(generator, Sequence) and use_multiprocessing and workers > 1:
            warnings.warn(UserWarning('For multiprocessing, use the instance of Sequence.'))
        
        out_labels = ['loss'] + [v.name for v in disc_ext.loss_functions] #?                                                   
        aggr_metrics = training_utils.MetricsAggregator(True, steps=self.hps['batch_step'])
        
        # Callbacks.
        callbacks = cbks.configure_callbacks(callbacks #?
                                            , disc_ext
                                            , do_validation=False
                                            , epochs=1
                                            , steps_per_epoch=self.hps['batch_step']
                                            , samples=self.hps['batch_step']
                                            , verbose=0
                                            , mode=ModeKeys.TEST)
        progbar = training_utils.get_progbar(disc_ext, 'steps')
        progbar.params = callbacks.params
        progbar.params['verbose'] = verbose
        
        # Evaluate.
        callbacks._call_begin_hook(ModeKeys.TEST)
        progbar.on_train_begin()
        
        disc_ext.reset_metrics()
        epochs_log= {}      
        callbacks.on_epoch_begin(0, epochs_log)
        progbar.on_epoch_begin(0, epochs_log)

        for k_i in range(self.hps['batch_step']):
            # Build batch logs.
            k_batch_logs = {'batch': self.hps['batch_step'] * k_i + 1, 'size': self.hps['batch_size']}
            callbacks._call_batch_hook(ModeKeys.TEST
                                                , 'begin'
                                                , self.hps['batch_step'] * k_i + 1
                                                , k_batch_logs)
            progbar.on_batch_begin(self.hps['batch_step'] * k_i + 1, k_batch_logs)
            
            inputs, outputs = gen_disc_ext_data_func(generator)
            outs = disc_ext.test_on_batch(inputs, outputs, reset_metrics=False) #?  
            del inputs, outputs
            outs = to_list(outs) #?
            outs_p = [np.asarray([v]) for v in outs]
            
            if k_i == 0:
                aggr_metrics.create(outs_p)
            else:
                aggr_metrics.aggregate(outs_p)
            
            metrics_names = ['loss'] + [v.name for v in disc_ext.loss_functions]                            
                
            #for l, o in zip(metrics_names, outs):
            #    k_batch_logs[l] = o
            k_batch_logs = cbks.make_logs(disc_ext, k_batch_logs, outs_p, ModeKeys.TEST)                        
            
            callbacks._call_batch_hook(ModeKeys.TEST
                                                , 'end'
                                                , self.hps['batch_step'] * k_i + 1
                                                , k_batch_logs)
            progbar.on_batch_end(self.hps['batch_step'] * k_i + 1, k_batch_logs)
            #print('\n', k_batch_logs)
                            
        aggr_metrics.finalize()
        
        # Make epochs log. ?
        outs = to_list(aggr_metrics.results)
        #for out_label, out in zip(out_labels, outs):
        #    epochs_log[out_label] = out
        epochs_log = cbks.make_logs(disc_ext
                                    , epochs_log
                                    , outs
                                    , ModeKeys.TEST)
                                                                      
        callbacks.on_epoch_end(0, epochs_log)
        progbar.on_epoch_end(0, epochs_log)
                    
        disc_ext._successful_loop_finish = True
        callbacks._call_end_hook(ModeKeys.TEST)

        return to_list(aggr_metrics.results)
    
    def _evaluate_gen_disc(self
                      , gen_disc
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
        gen_disc: ModelExt.
            Generator and discriminator composite model.
        generator: Generator
            Test data generator.
        verbose: Integer 
            Verbose mode (default=1).
        callback: list
            Callbacks (default=None).
        max_queue_size: Integer
            Maximum size for the generator queue (default: 10).
        class_weight: Numpy array. ?
            Class weight.
        workers: Integer
            Maximum number of processes to get samples (default: 1, 0: main thread).
        use_multiprocessing: Boolean
            Multi-processing flag (default: False).
.        
        Returns
        -------
        Training history.
            Tuple.
        """

        # Check exception.                    
        if not isinstance(generator, Sequence) and use_multiprocessing and workers > 1:
            warnings.warn(UserWarning('For multiprocessing, use the instance of Sequence.'))
        
        out_labels = ['loss'] + [v.name for v in gen_disc.loss_functions] #?                                                   
        aggr_metrics = training_utils.MetricsAggregator(True, steps=self.hps['batch_step'])

        # Callbacks.
        callbacks = cbks.configure_callbacks(callbacks #?
                                            , gen_disc
                                            , do_validation=False
                                            , epochs=1
                                            , steps_per_epoch=self.hps['batch_step']
                                            , samples=self.hps['batch_step']
                                            , verbose=0
                                            , mode=ModeKeys.TEST)
        progbar = training_utils.get_progbar(gen_disc, 'steps')
        progbar.params = callbacks.params
        progbar.params['verbose'] = verbose
        
        # Evaluate.
        callbacks._call_begin_hook(ModeKeys.TEST)
        progbar.on_train_begin()
        
        gen_disc.reset_metrics()
        epochs_log= {}      
        callbacks.on_epoch_begin(0, epochs_log)
        progbar.on_epoch_begin(0, epochs_log)

        for s_i in range(self.hps['batch_step']):                
            # Build batch logs.
            batch_logs = {'batch': s_i + 1, 'size': self.hps['batch_size']}
            callbacks._call_batch_hook(ModeKeys.TEST
                                                , 'begin'
                                                , s_i
                                                , batch_logs)
            progbar.on_batch_begin(self.hps['batch_step'] * s_i + 1, batch_logs)

            inputs, outputs = gen_gen_disc_data_func(generator)
            outs = gen_disc.test_on_batch(inputs, outputs, reset_metrics=False) #?  
            del inputs, outputs
            outs = to_list(outs)
            outs_p = [np.asarray([v]) for v in outs]
            
            if s_i == 0:
                aggr_metrics.create(outs_p)
            else:
                aggr_metrics.aggregate(outs_p)
            
            metrics_names = ['loss'] + [v.name for v in gen_disc.loss_functions]
            
            #for l, o in zip(metrics_names, outs): #?
            #    batch_logs[l] = o
            batch_logs = cbks.make_logs(gen_disc, batch_logs, outs_p, ModeKeys.TEST) 

            callbacks._call_batch_hook(ModeKeys.TEST
                                                , 'end'
                                                , s_i
                                                , batch_logs)
            #print('\n', batch_logs)
            progbar.on_batch_end(self.hps['batch_step'] * s_i + 1, batch_logs)
        
        aggr_metrics.finalize()
        
        # Make epochs log. ?
        outs = to_list(aggr_metrics.results)
        #for out_label, out in zip(out_labels, outs):
        #    epochs_log[out_label] = out
        epochs_log = cbks.make_logs(gen_disc
                                    , epochs_log
                                    , outs
                                    , ModeKeys.TEST)

        callbacks.on_epoch_end(0, epochs_log)
        progbar.on_epoch_end(0, epochs_log)
                           
        gen_disc._successful_loop_finish = True
        callbacks._call_end_hook(ModeKeys.TEST)

        return to_list(aggr_metrics.results)
        
    def generate(self, inputs, *args, **kwargs):
        """Generate.
        
        Parameters
        ----------
        inputs: Numpy array, list or tuple.
            Inputs.
        """
        inputs = inputs if isinstance(inputs, (list, tuple)) else [inputs]
        return self.gen.predict(inputs)
    
def compose_gan_with_mode(gen, disc, mode, multi_gpu=False, num_gpus=1):
    """Compose the GAN model with mode.
    
    Parameters
    ----------
    gan: ModelExt.
        Generator model.
    disc: ModelExt.
        Discriminator model.
    mode: Integer.
        GAN composing mode.
    """
    assert isinstance(gen, (Model, ModelExt)) and isinstance(disc, (Model, ModelExt))
            
    if mode == STYLE_GAN_REGULAR or mode == LSGAN:
        # Compose gan.                    
        # Compose disc_ext.
        # disc.
        x_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in disc.inputs]  
        x_outputs = [disc(x_inputs)] if len(disc.outputs) == 1 else disc(x_inputs) #? 
        
        # gen and disc.
        z_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in gen.inputs] 
               
        gen.trainable = False
        for layer in gen.layers: layer.trainable = False
        z_outputs = [gen(z_inputs)] if len(gen.outputs) == 1 else gen(z_inputs)
        
        disc.trainable = True
        for layer in disc.layers: layer.trainable = True
        x2_outputs = [disc(z_outputs)] if len(disc.outputs) == 1 else disc(z_outputs)
        
        disc_ext = ModelExt(inputs=x_inputs + z_inputs
                            , outputs=x_outputs + x2_outputs
                            , name='disc_ext')
        if multi_gpu:
            disc_ext = multi_gpu_model(disc_ext, gpus=num_gpus, name='disc_ext') # Name?   
                
        # Compose gen_disc.
        z_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in gen.inputs] 
        
        gen.trainable = True
        for layer in gen.layers: layer.trainable = True    
        z_outputs = [gen(z_inputs)] if len(gen.outputs) == 1 else gen(z_inputs)
        
        disc.trainable = False
        for layer in disc.layers: layer.trainable = False
        z_p_outputs = [disc(z_outputs)] if len(disc.outputs) == 1 else disc(z_outputs)

        gen_disc = ModelExt(inputs=z_inputs
                            , outputs=z_p_outputs
                            , name='gen_disc')
        if multi_gpu:
            gen_disc = multi_gpu_model(gen_disc, gpus=num_gpus, name='gen_disc')
    elif mode == STYLE_GAN_WGAN_GP:
        # Compose gan.                    
        # Compose disc_ext.
        # disc.
        x_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in disc.inputs]  
        x_outputs = [disc(x_inputs)] if len(disc.outputs) == 1 else disc(x_inputs) #? 
        
        # gen and disc.
        z_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in gen.inputs] 
               
        gen.trainable = False
        for layer in gen.layers: layer.trainable = False
        z_outputs = [gen(z_inputs)] if len(gen.outputs) == 1 else gen(z_inputs)
        
        disc.trainable = True
        for layer in disc.layers: layer.trainable = True
        x2_outputs = [disc(z_outputs)] if len(disc.outputs) == 1 else disc(z_outputs)
        
        x3_inputs = [[tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in disc.inputs][0]]
        x3_outputs = [disc(x3_inputs + [x_inputs[1]])]
        
        disc_ext = ModelExt(inputs=x_inputs + z_inputs + x3_inputs
                            , outputs=x_outputs + x2_outputs + x3_outputs
                            , name='disc_ext')
        if multi_gpu:
            disc_ext = multi_gpu_model(disc_ext, gpus=num_gpus, name='disc_ext') # Name?   
                
        # Compose gen_disc.
        z_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in gen.inputs] 
        
        gen.trainable = True
        for layer in gen.layers: layer.trainable = True    
        z_outputs = [gen(z_inputs)] if len(gen.outputs) == 1 else gen(z_inputs)
        
        disc.trainable = False
        for layer in disc.layers: layer.trainable = False
        z_p_outputs = [disc(z_outputs)] if len(disc.outputs) == 1 else disc(z_outputs)

        gen_disc = ModelExt(inputs=z_inputs
                            , outputs=z_p_outputs
                            , name='gen_disc')
        if multi_gpu:
            gen_disc = multi_gpu_model(gen_disc, gpus=num_gpus, name='gen_disc')
    elif mode == STYLE_GAN_SOFTPLUS_INVERSE_R1_GP:
        # Compose gan.                    
        # Compose disc_ext.
        # disc.
        x_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in disc.inputs]  
        x_outputs = [disc(x_inputs)] if len(disc.outputs) == 1 else disc(x_inputs) #? 
        
        # gen and disc.
        z_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in gen.inputs] 
               
        gen.trainable = False
        for layer in gen.layers: layer.trainable = False
        z_outputs = [gen(z_inputs)] if len(gen.outputs) == 1 else gen(z_inputs)
        
        disc.trainable = True
        for layer in disc.layers: layer.trainable = True
        x2_outputs = [disc(z_outputs)] if len(disc.outputs) == 1 else disc(z_outputs)
                
        disc_ext = ModelExt(inputs=x_inputs + z_inputs
                            , outputs=x_outputs + x_outputs + x2_outputs
                            , name='disc_ext')
        if multi_gpu:
            disc_ext = multi_gpu_model(disc_ext, gpus=num_gpus, name='disc_ext') # Name?   
                
        # Compose gen_disc.
        z_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in gen.inputs] 
        
        gen.trainable = True
        for layer in gen.layers: layer.trainable = True    
        z_outputs = [gen(z_inputs)] if len(gen.outputs) == 1 else gen(z_inputs)
        
        disc.trainable = False
        for layer in disc.layers: layer.trainable = False
        z_p_outputs = [disc(z_outputs)] if len(disc.outputs) == 1 else disc(z_outputs)

        gen_disc = ModelExt(inputs=z_inputs
                            , outputs=z_p_outputs
                            , name='gen_disc')
        if multi_gpu:
            gen_disc = multi_gpu_model(gen_disc, gpus=num_gpus, name='gen_disc')   
    elif mode == PIX2PIX_GAN:
        # Compose gan.                    
        # Compose disc_ext.
        # disc.
        x_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in disc.inputs]  
        x_outputs = [disc(x_inputs)] if len(disc.outputs) == 1 else disc(x_inputs) #? 
        
        # gen and disc.
        z_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in gen.inputs]
               
        gen.trainable = False
        for layer in gen.layers: layer.trainable = False
        z_outputs = [gen(z_inputs)] if len(gen.outputs) == 1 else gen(z_inputs)
        
        disc.trainable = True
        for layer in disc.layers: layer.trainable = True
        
        # Get condition inputs.
        cond_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in disc.inputs \
                         if 'cond' in t.name]        
                        
        x2_outputs = [disc(cond_inputs + z_outputs)] \
            if len(disc.outputs) == 1 else disc(cond_inputs + z_outputs)
        
        disc_ext = ModelExt(inputs=x_inputs + z_inputs + cond_inputs
                            , outputs=x_outputs + x2_outputs
                            , name='disc_ext')
        if multi_gpu:
            disc_ext = multi_gpu_model(disc_ext, gpus=num_gpus, name='disc_ext') 
                
        # Compose gen_disc.
        z_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in gen.inputs] 
        
        gen.trainable = True
        for layer in gen.layers: layer.trainable = True    
        z_outputs = [gen(z_inputs)] if len(gen.outputs) == 1 else gen(z_inputs)
        
        disc.trainable = False
        for layer in disc.layers: layer.trainable = False

        # Get condition inputs.
        cond_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in disc.inputs \
                        if 'cond' in t.name]
         
        z_p_outputs = [disc(cond_inputs + z_outputs)] \
            if len(disc.outputs) == 1 else disc(cond_inputs + z_outputs)

        gen_disc = ModelExt(inputs=z_inputs + cond_inputs
                            , outputs=z_p_outputs + z_outputs 
                            , name='gen_disc')
        if multi_gpu:
            gen_disc = multi_gpu_model(gen_disc, gpus=num_gpus)
    elif mode == CYCLE_GAN:
        # TODO
        pass
    else:
        ValueError('mode is not valid.')
    
    return disc_ext, gen_disc            