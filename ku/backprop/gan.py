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
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import Sequence, GeneratorEnqueuer, OrderedEnqueuer
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Lambda
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError

from tensorflow.python.keras.utils.generic_utils import to_list, CustomObjectScope
from tensorflow.python.keras.utils.data_utils import iter_sequence_infinite
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.engine import training_utils
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
        conf: Dict.
            Configuration.
        """
        self.conf = conf #?
        
        if self.conf['model_loading']:
            if not hasattr(self, 'custom_objects'):
                raise RuntimeError('Before models, custom_objects must be created.')
                                                          
            self.custom_objects['ModelExt'] = ModelExt                                                          

            # gen_disc.
            self.gen_disc = load_model(self.GEN_DISC_PATH
                                       , custom_objects=self.custom_objects
                                       , compile=False) #?

            # gen, disc.
            self.gen = self.gen_disc.get_layer('gen')
            self.disc = self.gen_disc.get_layer('disc')

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
                         , run_eagerly=True) # run_eagerly?
        
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
                          , callbacks_disc_ext_raw=None
                          , callbacks_gen_disc_raw=None
                          , validation_data_gen=None  # ?
                          , validation_steps=None
                          , validation_freq=1  # ?
                          , class_weight=None  # ?
                          , max_queue_size=10
                          , workers=1
                          , use_multiprocessing=False  # ?
                          , shuffle=True
                          , initial_epoch=0
                          , save_f=True):  # ?
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
            callback_disc_ext_raw: list.
                disc_ext callbacks (default=None).
            callback_gen_disc_raw: list.
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

            '''
            _keras_api_gauge.get_cell('fit').set(True)
            # Legacy graph support is contained in `training_v1.Model`.
            version_utils.disallow_legacy_graph('Model', 'fit')
            self._assert_compile_was_called()
            self._check_call_args('fit')
            _disallow_inside_tf_function('fit')
            '''

            # Check exception.
            do_validation = bool(validation_data_gen)
            if do_validation:
                assert hasattr(validation_data_gen, 'next') or \
                       hasattr(validation_data_gen, '__next') or \
                       isinstance(validation_data_gen, Sequence)

                if not isinstance(validation_data_gen, Sequence):
                    assert validation_steps  # ?

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
                                                      , use_multiprocessing=use_multiprocessing)  # shuffle?
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
                if not isinstance(callbacks_disc_ext_raw, cbks.CallbackList):
                    callbacks_disc_ext = cbks.CallbackList(callbacks_disc_ext_raw
                                                           , add_history=True
                                                           , add_progbar=verbose != 0
                                                           , model=self.disc_ext
                                                           , verbose=verbose
                                                           , epochs=self.hps['epochs']
                                                           , steps=self.hps['batch_step'] * self.hps['disc_k_step'])
                else:
                    callbacks_disc_ext = callbacks_disc_ext_raw

                # gen_disc.
                if not isinstance(callbacks_disc_ext_raw, cbks.CallbackList):
                    callbacks_gen_disc = cbks.CallbackList(callbacks_gen_disc_raw
                                                           , add_history=True
                                                           , add_progbar=verbose != 0
                                                           , model=self.gen_disc
                                                           , verbose=verbose
                                                           , epochs=self.hps['epochs']
                                                           , steps=self.hps['batch_step'])
                else:
                    callbacks_gen_disc = callbacks_gen_disc_raw

                # Train.
                self.disc_ext.stop_training = False
                self.disc_ext._train_counter.assign(0)
                self.gen_disc.stop_training = False
                self.gen_disc._train_counter.assign(0)
                disc_ext_training_logs = None
                gen_disc_training_logs = None

                callbacks_disc_ext.on_train_begin()
                callbacks_gen_disc.on_train_begin()

                initial_epoch = self.disc_ext._maybe_load_initial_epoch_from_ckpt(initial_epoch)  # ?

                pre_e_i = -1
                for e_i in range(initial_epoch, self.hps['epochs']):
                    if callbacks_disc_ext.model.stop_training or callbacks_gen_disc.model.stop_training:
                        break

                    self.disc_ext.reset_metrics()  # ?
                    self.gen_disc.reset_metrics()

                    epochs_log_disc_ext = {}
                    epochs_log_gen_disc = {}

                    callbacks_disc_ext.on_epoch_begin(e_i, epochs_log_disc_ext)
                    callbacks_gen_disc.on_epoch_begin(e_i, epochs_log_gen_disc)

                    for s_i in range(self.hps['batch_step']):
                        for k_i in range(self.hps['disc_k_step']):
                            step = self.hps['disc_k_step'] * s_i + k_i - 1  # ?
                            with trace.Trace('TraceContext'
                                    , graph_type='train'
                                    , epoch_num=e_i
                                    , step_num=step
                                    , batch_size=self.hps['batch_size']):
                                callbacks_disc_ext.on_train_batch_begin(step)

                                inputs, outputs = gen_disc_ext_data_fun(output_generator)

                                self.gen.trainable = False
                                for layer in self.gen.layers: layer.trainable = False

                                self.disc.trainable = True
                                for layer in self.disc.layers: layer.trainable = True

                                disc_ext_step_logs = self.disc_ext.train_on_batch(inputs
                                                                                  , outputs
                                                                                  , class_weight=class_weight
                                                                                  , reset_metrics=False
                                                                                  , return_dict=True)  # ?
                                del inputs, outputs

                                end_step = step + 1
                                callbacks_disc_ext.on_train_batch_end(end_step, disc_ext_step_logs)

                        step = s_i - 1  # ?
                        with trace.Trace('TraceContext'
                                , graph_type='train'
                                , epoch_num=e_i
                                , step_num=step
                                , batch_size=self.hps['batch_size']):

                            inputs, outputs = gen_gen_disc_data_fun(output_generator)

                            self.gen.trainable = True
                            for layer in self.gen.layers: layer.trainable = True

                            self.disc.trainable = False
                            for layer in self.disc.layers: layer.trainable = False

                            gen_disc_step_logs = self.gen_disc.train_on_batch(inputs
                                                                              , outputs
                                                                              , class_weight=class_weight
                                                                              , reset_metrics=False
                                                                              , return_dict=True)
                            del inputs, outputs

                            end_step = step + 1
                            callbacks_gen_disc.on_train_batch_end(end_step, gen_disc_step_logs)

                    disc_ext_epoch_logs = copy.copy(disc_ext_step_logs)  # ?
                    gen_disc_epoch_logs = copy.copy(gen_disc_step_logs)  # ?

                    # Do validation.
                    if do_validation:  # ?
                        if e_i % validation_freq == 0:  # ?
                            # disc_ext.
                            val_outs_disc_ext = self._evaluate_disc_ext(self.disc_ext
                                                                        , val_generator  # ?
                                                                        , gen_disc_ext_data_fun
                                                                        , callbacks_raw=None  # callbacks_disc_ext
                                                                        , workers=1
                                                                        , verbose=1)

                            # gen_disc.
                            val_outs_gen_disc = self._evaluate_gen_disc(self.gen_disc
                                                                        , val_generator
                                                                        , gen_gen_disc_data_fun
                                                                        , callbacks_raw=None  # callbacks_gen_disc
                                                                        , workers=1
                                                                        , verbose=1)
                            # Make epochs logs.
                            epochs_log_disc_ext.update(val_outs_disc_ext)
                            epochs_log_gen_disc.update(val_outs_gen_disc)

                    callbacks_disc_ext.on_epoch_end(e_i, epochs_log_disc_ext)
                    callbacks_gen_disc.on_epoch_end(e_i, epochs_log_gen_disc)
                    disc_ext_training_logs = epochs_log_disc_ext
                    gen_disc_training_logs = epochs_log_gen_disc

                    if save_f:
                        self.save_gan_model()

                    pre_e_i = e_i

                callbacks_disc_ext.on_train_end(logs=disc_ext_training_logs)
                callbacks_gen_disc.on_train_end(logs=gen_disc_training_logs)
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
                                        , validation_data_gen=None  # ?
                                        , validation_steps=None
                                        , validation_freq=1  # ?
                                        , class_weight=None  # ?
                                        , max_queue_size=10
                                        , workers=1
                                        , use_multiprocessing=False  # ?
                                        , shuffle=True
                                        , initial_epoch=0
                                        , save_f=True):  # ?
            """Train the GAN model with the generator progressively.

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

            '''
            _keras_api_gauge.get_cell('fit').set(True)
            # Legacy graph support is contained in `training_v1.Model`.
            version_utils.disallow_legacy_graph('Model', 'fit')
            self._assert_compile_was_called()
            self._check_call_args('fit')
            _disallow_inside_tf_function('fit')
            '''

            # Check exception.
            do_validation = bool(validation_data_gen)
            if do_validation:
                assert hasattr(validation_data_gen, 'next') or \
                       hasattr(validation_data_gen, '__next') or \
                       isinstance(validation_data_gen, Sequence)

                if not isinstance(validation_data_gen, Sequence):
                    assert validation_steps  # ?

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
                                                      , use_multiprocessing=use_multiprocessing)  # shuffle?
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
                if not isinstance(callbacks_disc_ext_raw, cbks.CallbackList):
                    callbacks_disc_ext = cbks.CallbackList(callbacks_disc_ext_raw
                                                           , add_history=True
                                                           , add_progbar=verbose != 0
                                                           , model=self.disc_ext
                                                           , verbose=verbose
                                                           , epochs=self.hps['epochs']
                                                           , steps=self.hps['batch_step'] * self.hps['disc_k_step'])
                else:
                    callbacks_disc_ext = callbacks_disc_ext_raw

                # gen_disc.
                if not isinstance(callbacks_disc_ext_raw, cbks.CallbackList):
                    callbacks_gen_disc = cbks.CallbackList(callbacks_gen_disc_raw
                                                           , add_history=True
                                                           , add_progbar=verbose != 0
                                                           , model=self.gen_disc
                                                           , verbose=verbose
                                                           , epochs=self.hps['epochs']
                                                           , steps=self.hps['batch_step'])
                else:
                    callbacks_gen_disc = callbacks_gen_disc_raw

                # Train.
                self.disc_ext.model.stop_training = False
                self.disc_ext._train_counter.assign(0)
                self.gen_disc.model.stop_training = False
                self.gen_disc._train_counter.assign(0)
                disc_ext_training_logs = None
                gen_disc_training_logs = None

                callbacks_disc_ext.on_train_begin()
                callbacks_gen_disc.on_train_begin()

                initial_epoch = self.disc_ext._maybe_load_initial_epoch_from_ckpt(
                    initial_epoch)  # ?

                pre_e_i = -1
                for e_i in range(initial_epoch, self.hps['epochs']):
                    if callbacks_disc_ext.model.stop_training or callbacks_gen_disc.model.stop_training:
                        break

                    self.disc_ext.reset_metrics()  # ?
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
                            step = self.hps['disc_k_step'] * s_i + k_i - 1  # ?
                            with trace.Trace('TraceContext'
                                    , graph_type='train'
                                    , epoch_num=e_i
                                    , step_num=step
                                    , batch_size=self.hps['batch_size']):
                                callbacks_disc_ext.on_train_batch_begin(step)

                                inputs, outputs = gen_disc_ext_data_fun(output_generator)

                                self.gen.trainable = False
                                for layer in self.gen.layers: layer.trainable = False

                                self.disc.trainable = True
                                for layer in self.disc.layers: layer.trainable = True

                                disc_ext_step_logs = partial_disc_ext.train_on_batch(inputs
                                                                                     , outputs
                                                                                     , class_weight=class_weight
                                                                                     , reset_metrics=False
                                                                                     , return_dict=True)  # ?
                                del inputs, outputs

                                end_step = step + 1
                                callbacks_disc_ext.on_train_batch_end(end_step, disc_ext_step_logs)

                        step = s_i - 1  # ?
                        with trace.Trace('TraceContext'
                                , graph_type='train'
                                , epoch_num=e_i
                                , step_num=step
                                , batch_size=self.hps['batch_size']):

                            inputs, outputs = gen_gen_disc_data_fun(output_generator)

                            self.gen.trainable = True
                            for layer in self.gen.layers: layer.trainable = True

                            self.disc.trainable = False
                            for layer in self.disc.layers: layer.trainable = False

                            gen_disc_step_logs = partial_gen_disc.train_on_batch(inputs
                                                                                 , outputs
                                                                                 , class_weight=class_weight
                                                                                 , reset_metrics=False
                                                                                 , return_dict=True)
                            del inputs, outputs

                            end_step = step + 1
                            callbacks_gen_disc.on_train_batch_end(end_step, gen_disc_step_logs)

                    disc_ext_epoch_logs = copy.copy(disc_ext_step_logs)  # ?
                    gen_disc_epoch_logs = copy.copy(gen_disc_step_logs)  # ?

                    # Do validation.
                    if do_validation:  # ?
                        if e_i % validation_freq == 0:  # ?
                            # disc_ext.
                            val_outs_disc_ext = self._evaluate_disc_ext(self.disc_ext
                                                                        , val_generator  # ?
                                                                        , gen_disc_ext_data_fun
                                                                        , callbacks_raw=None  # callbacks_disc_ext
                                                                        , workers=1
                                                                        , verbose=1)

                            # gen_disc.
                            val_outs_gen_disc = self._evaluate_gen_disc(self.gen_disc
                                                                        , val_generator
                                                                        , gen_gen_disc_data_fun
                                                                        , callbacks_raw=None  # callbacks_gen_disc
                                                                        , workers=1
                                                                        , verbose=1)
                            # Make epochs logs.
                            epochs_log_disc_ext.update(val_outs_disc_ext)
                            epochs_log_gen_disc.update(val_outs_gen_disc)

                    callbacks_disc_ext.on_epoch_end(e_i, epochs_log_disc_ext)
                    callbacks_gen_disc.on_epoch_end(e_i, epochs_log_gen_disc)
                    disc_ext_training_logs = epochs_log_disc_ext
                    gen_disc_training_logs = epochs_log_gen_disc

                    if save_f:
                        self.save_gan_model()

                    pre_e_i = e_i

                callbacks_disc_ext.on_train_end(logs=disc_ext_training_logs)  # progress bar?
                callbacks_gen_disc.on_train_end(logs=gen_disc_training_logs)
            finally:
                try:
                    if enq:
                        enq.stop()
                finally:
                    if val_enq:
                        val_enq.stop()

            return self.disc_ext.history, self.gen_disc.history

        def _evaluate_disc_ext(self
                               , disc_ext
                               , generator
                               , gen_disc_ext_data_func
                               , verbose=1
                               , callbacks_raw=None
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
            Evaluating result.
                Dictionary.
            """

            '''
            _keras_api_gauge.get_cell('evaluate').set(True)
            version_utils.disallow_legacy_graph('Model', 'evaluate')
            self._assert_compile_was_called()
            self._check_call_args('evaluate')
            _disallow_inside_tf_function('evaluate')
            '''

            # Check exception.
            if not isinstance(generator, Sequence) and use_multiprocessing and workers > 1:
                warnings.warn(UserWarning('For multiprocessing, use the instance of Sequence.'))

            # Callbacks.
            if not isinstance(callbacks_raw, cbks.CallbackList):
                callbacks = cbks.CallbackList(callbacks_raw
                                              , add_history=True
                                              , add_progbar=verbose != 0
                                              , model=disc_ext
                                              , verbose=verbose
                                              , epochs=1
                                              , steps=self.hps['batch_step'])
            else:
                callbacks = callbacks_raw

            # Evaluate.
            logs = {}
            disc_ext._test_counter.assign(0)

            callbacks.on_test_begin()
            disc_ext.reset_metrics()

            for k_i in range(self.hps['batch_step']):  # ?
                step = k_i - 1
                with trace.Trace('TraceContext', graph_type='test', step_num=step):
                    callbacks.on_test_batch_begin(step)

                    inputs, outputs = gen_disc_ext_data_func(generator)
                    tmp_logs = disc_ext.test_on_batch(inputs
                                                      , outputs
                                                      , reset_metrics=False
                                                      , return_dict=True)
                    del inputs, outputs

                    logs = tmp_logs  # No error, now safe to assign to logs.
                    end_step = step + 1
                    callbacks.on_test_batch_end(end_step, logs)

            logs = tf_utils.to_numpy_or_python_type(logs)
            callbacks.on_test_end(logs=logs)

            return logs

        def _evaluate_gen_disc(self
                               , gen_disc
                               , generator
                               , gen_gen_disc_data_func
                               , verbose=1
                               , callbacks_raw=None
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
            Evaluating result.
                Dictionary.
            """

            '''
            _keras_api_gauge.get_cell('evaluate').set(True)
            version_utils.disallow_legacy_graph('Model', 'evaluate')
            self._assert_compile_was_called()
            self._check_call_args('evaluate')
            _disallow_inside_tf_function('evaluate')
            '''

            # Check exception.
            if not isinstance(generator, Sequence) and use_multiprocessing and workers > 1:
                warnings.warn(UserWarning('For multiprocessing, use the instance of Sequence.'))

            # Callbacks.
            if not isinstance(callbacks_raw, cbks.CallbackList):
                callbacks = cbks.CallbackList(callbacks_raw
                                              , add_history=True
                                              , add_progbar=verbose != 0
                                              , model=gen_disc
                                              , verbose=verbose
                                              , epochs=1
                                              , steps=self.hps['batch_step'])
            else:
                callbacks = callbacks_raw

            # Evaluate.
            logs = {}
            gen_disc._test_counter.assign(0)

            callbacks.on_test_begin()
            gen_disc.reset_metrics()

            for s_i in range(self.hps['batch_step']):
                step = s_i - 1
                with trace.Trace('TraceContext', graph_type='test', step_num=step):
                    callbacks.on_test_batch_begin(s_i)

                    inputs, outputs = gen_gen_disc_data_func(generator)
                    tmp_logs = gen_disc.test_on_batch(inputs
                                                      , outputs
                                                      , reset_metrics=False
                                                      , return_dict=True)
                    del inputs, outputs

                    logs = tmp_logs  # No error, now safe to assign to logs.
                    end_step = step + 1
                    callbacks.on_test_batch_end(end_step, logs)

            logs = tf_utils.to_numpy_or_python_type(logs)
            callbacks.on_test_end(logs=logs)

            return logs

    def save_gan_model(self):
        """Save the GAN model."""
        assert hasattr(self, 'disc_ext') and hasattr(self, 'gen_disc')
        
        with CustomObjectScope(self.custom_objects):
            self.disc_ext.save(self.DISC_EXT_PATH, save_format='h5')
            self.gen_disc.save(self.GEN_DISC_PATH, save_format='h5')

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
        x2_outputs = [disc(z_outputs + [z_inputs[1]])] if len(disc.outputs) == 1 else disc(z_outputs)
        
        disc_ext = ModelExt(inputs=x_inputs + z_inputs
                            , outputs=x_outputs + x2_outputs
                            , name='disc_ext')

        # Compose gen_disc.
        z_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in gen.inputs] 
        
        gen.trainable = True
        for layer in gen.layers: layer.trainable = True    
        z_outputs = [gen(z_inputs)] if len(gen.outputs) == 1 else gen(z_inputs)
        
        disc.trainable = False
        for layer in disc.layers: layer.trainable = False
        z_p_outputs = [disc(z_outputs + [z_inputs[1]])] if len(disc.outputs) == 1 else disc(z_outputs)

        gen_disc = ModelExt(inputs=z_inputs
                            , outputs=z_p_outputs
                            , name='gen_disc')
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
        x2_outputs = [disc(z_outputs + [z_inputs[1]])] if len(disc.outputs) == 1 else disc(z_outputs)
        
        x3_inputs = [[tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in disc.inputs][0]]
        x3_outputs = [disc(x3_inputs + [x_inputs[1]])]
        
        disc_ext = ModelExt(inputs=x_inputs + z_inputs + x3_inputs
                            , outputs=x_outputs + x2_outputs + x3_outputs
                            , name='disc_ext')
                
        # Compose gen_disc.
        z_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in gen.inputs] 
        
        gen.trainable = True
        for layer in gen.layers: layer.trainable = True    
        z_outputs = [gen(z_inputs)] if len(gen.outputs) == 1 else gen(z_inputs)
        
        disc.trainable = False
        for layer in disc.layers: layer.trainable = False
        z_p_outputs = [disc(z_outputs + [z_inputs[1]])] if len(disc.outputs) == 1 else disc(z_outputs)

        gen_disc = ModelExt(inputs=z_inputs
                            , outputs=z_p_outputs
                            , name='gen_disc')
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
        x2_outputs = [disc(z_outputs + [z_inputs[1]])] if len(disc.outputs) == 1 else disc(z_outputs)
                
        disc_ext = ModelExt(inputs=x_inputs + z_inputs
                            , outputs=x_outputs + x_outputs + x2_outputs
                            , name='disc_ext')

        # Compose gen_disc.
        z_inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in gen.inputs] 
        
        gen.trainable = True
        for layer in gen.layers: layer.trainable = True    
        z_outputs = [gen(z_inputs)] if len(gen.outputs) == 1 else gen(z_inputs)
        
        disc.trainable = False
        for layer in disc.layers: layer.trainable = False
        z_p_outputs = [disc(z_outputs + [z_inputs[1]])] if len(disc.outputs) == 1 else disc(z_outputs)

        gen_disc = ModelExt(inputs=z_inputs
                            , outputs=z_p_outputs
                            , name='gen_disc')
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
    else:
        ValueError('mode is not valid.')
    
    return disc_ext, gen_disc            