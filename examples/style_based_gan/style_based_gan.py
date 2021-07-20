"""
Created on 2019. 6. 19.

@author: Inwoo Chung (gutomitai@gmail.com)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import platform
import json
import glob
import shutil
import warnings

import numpy as np
import pandas as pd
from skimage.io import imread, imsave
import cv2 as cv
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Embedding, Flatten, Dropout
from tensorflow.keras.layers import LeakyReLU, Activation, AveragePooling2D, UpSampling2D, Concatenate
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import optimizers
from tensorflow.keras.utils import Sequence, GeneratorEnqueuer, OrderedEnqueuer

from tensorflow.python.profiler import trace
from tensorflow.python.keras.utils.generic_utils import to_list, CustomObjectScope
from tensorflow.python.keras.utils.data_utils import iter_sequence_infinite
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys #?

''' TF 2.4.
from tensorflow.keras.utils.generic_utils import to_list, CustomObjectScope
from tensorflow.keras.utils.data_utils import iter_sequence_infinite
from tensorflow.keras import callbacks as cbks
from tensorflow.keras.engine import training_utils
from tensorflow.keras.utils.mode_keys import ModeKeys #?
'''

from ku.engine_ext import ModelExt
from ku.backprop import AbstractGAN
from ku.layer_ext import AdaptiveINWithStyle, TruncationTrick, StyleMixingRegularization
from ku.layer_ext import EqualizedLRDense, EqualizedLRConv2D
from ku.layer_ext import FusedEqualizedLRConv2DTranspose, BlurDepthwiseConv2D, FusedEqualizedLRConv2D
from ku.layer_ext import MinibatchStddevConcat
from ku.image_utils import resize
import ku.backprop.gan as gan
from ku.backprop.gan import compose_gan_with_mode

#os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# Constants.
DEBUG = True
MULTI_GPU = False
NUM_GPUS = 4

MODE_TRAINING = 0
MODE_VALIDATION = 1

class StyleGAN(AbstractGAN):
    """Style based GAN."""

    def __init__(self, conf):
        """
        Parameters
        ----------
        conf: Dict.
            Configuration.
        """
        self.conf = conf #?
        self.raw_data_path = self.conf['raw_data_path']
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']
        
        self.gen_hps = conf['gen_hps']
        self.gen_nn_arch = conf['gen_nn_arch']

        self.map_hps = conf['map_hps']
        self.map_nn_arch = conf['map_nn_arch']
        
        self.disc_hps = conf['disc_hps']
        self.disc_nn_arch = conf['disc_nn_arch']
        
        self.disc_ext_hps = conf['disc_ext_hps']
        self.gen_disc_hps = conf['gen_disc_hps']
                
        # Create models.
        if self.conf['model_loading'] != True:
            # Create generator and discriminator.

            self._create_generator()
            self._create_discriminator()
                        
            # Compose.
            self.compose_gan_with_mode(gan.STYLE_GAN_SOFTPLUS_INVERSE_R1_GP) #?
            
            # Compile.
            # Configure optimizers and losses.
            disc_ext_opt = optimizers.Adam(lr=self.disc_ext_hps['lr']
                                    , beta_1=self.disc_ext_hps['beta_1']
                                    , beta_2=self.disc_ext_hps['beta_2']
                                    , decay=self.disc_ext_hps['decay'])
            gen_disc_opt = optimizers.Adam(lr=self.gen_disc_hps['lr']
                                    , beta_1=self.gen_disc_hps['beta_1']
                                    , beta_2=self.gen_disc_hps['beta_2']
                                    , decay=self.gen_disc_hps['decay'])
            loss_conf = gan.get_loss_conf(self.hps
                                          , gan.LOSS_CONF_TYPE_NON_SATURATION_SOFTPLUS_R1_GP
                                          , model=self.disc_ext
                                          , input_variable_orders=[0])
            
            self.compile(disc_ext_opt
                         , loss_conf['disc_ext_losses']
                         , loss_conf['disc_ext_loss_weights']
                         , gen_disc_opt
                         , loss_conf['gen_disc_losses']
                         , loss_conf['gen_disc_loss_weights'])
                           
        self.custom_objects = {'AdaptiveINWithStyle': AdaptiveINWithStyle
                           , 'TruncationTrick': TruncationTrick
                           , 'StyleMixingRegularization': StyleMixingRegularization
                           , 'EqualizedLRDense': EqualizedLRDense
                           , 'EqualizedLRConv2D': EqualizedLRConv2D
                           , 'FusedEqualizedLRConv2DTranspose': FusedEqualizedLRConv2DTranspose
                           , 'BlurDepthwiseConv2D': BlurDepthwiseConv2D
                           , 'FusedEqualizedLRConv2D': FusedEqualizedLRConv2D
                           , "MinibatchStddevConcat": MinibatchStddevConcat
                           , 'resize': resize}
      
        super(StyleGAN, self).__init__(conf) #?
        
        if self.conf['model_loading']:
            # Compose.
            self.compose_gan_with_mode(gan.STYLE_GAN_SOFTPLUS_INVERSE_R1_GP) #?
            
            # Compile.
            # Configure optimizers and losses.
            disc_ext_opt = optimizers.Adam(lr=self.disc_ext_hps['lr']
                                    , beta_1=self.disc_ext_hps['beta_1']
                                    , beta_2=self.disc_ext_hps['beta_2']
                                    , decay=self.disc_ext_hps['decay'])
            gen_disc_opt = optimizers.Adam(lr=self.gen_disc_hps['lr']
                                    , beta_1=self.gen_disc_hps['beta_1']
                                    , beta_2=self.gen_disc_hps['beta_2']
                                    , decay=self.gen_disc_hps['decay'])
            loss_conf = gan.get_loss_conf(self.hps
                                          , gan.LOSS_CONF_TYPE_NON_SATURATION_SOFTPLUS_R1_GP
                                          , model=self.disc_ext
                                          , input_variable_orders=[0])
            
            self.compile(disc_ext_opt
                         , loss_conf['disc_ext_losses']
                         , loss_conf['disc_ext_loss_weights']
                         , gen_disc_opt
                         , loss_conf['gen_disc_losses']
                         , loss_conf['gen_disc_loss_weights'])
                    
    def _cal_num_chs(self, layer_idx):
        """Calculate the number of channels for each synthesis layer.
        
        Parameters
        ----------
        layer_idx: Integer
            Layer index.
        
        Returns:
            Number of channels for each layer.
                Integer
        """
        return int(np.min([int(self.hps['ch_base']) / (2.0 ** layer_idx)
                           , self.hps['max_ch']]))
        
    def _create_generator(self):
        """Create generator."""
        
        # Design generator.
        res_log2 = int(np.log2(self.nn_arch['resolution']))
        assert self.nn_arch['resolution'] == 2 ** res_log2 and self.nn_arch['resolution'] >= 4 #?
        self.nn_arch['num_layers'] = res_log2 * 2 - 2
        
        # Mapping network.
        self._create_mapping_net()
        
        # Inputs.
        inputs1 = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in self.map.inputs]
        
        #if self.nn_arch['label_usage']:
        #    output2 = Lambda(lambda x: x)(inputs1[1]) 
        
        # Disentangled latent.
        dlatents1 = self.map(inputs1)

        # Style mixing regularization.
        if self.nn_arch['label_usage']:
            inputs2 = [Input(shape=K.int_shape(inputs1[0])[1:]), inputs1[1]] # Normal random input.
        else:
            inputs2 = Input(shape=K.int_shape(inputs1[0])[1:]) # Normal random input.
        
        dlatents2 = self.map(inputs2)
        
        #dlatents = Lambda(lambda x: x[0])([dlatents1, dlatents2])
        
        dlatents = StyleMixingRegularization(mixing_prob=self.hps['mixing_prob'])([dlatents1, dlatents2])
        
        # Truncation trick.
        dlatents = TruncationTrick(psi=self.hps['trunc_psi']
                 , cutoff=self.hps['trunc_cutoff']
                 , momentum=self.hps['trunc_momentum'])(dlatents)

        # Design the model according to the final image resolution.                
        # The first constant input layer.
        res = 2
        layer_idx = 0
        
        # Input variables.
        num_chs = self._cal_num_chs(res - 1)
        def _gen_first_layer_block(dlatents_p): #?
            x = K.variable(np.ones(shape=(1, 4, 4, num_chs)))
            
            x = StyleGAN._apply_noise_layer(x)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + 1e-8)
    
            # Check exception.
            axis = -1            
            if axis < 0:
                assert K.ndim(x) + axis >= 0
                axis = K.ndim(x) + axis #?
                
            reduce_axes = tuple([i for i in range(1, K.ndim(x)) if i != axis])
            
            c = x # Content image tensor.
            s = dlatents_p # Style dlatent tensor.
            
            # Calculate mean and variance.
            c_mean = K.mean(c, axis=reduce_axes, keepdims=True)
            c_std = K.std(c, axis=reduce_axes, keepdims=True) + 1e-8
            s = K.reshape(s, [-1, 2, 1, 1, c.shape[-1]]) #?
            
            return (s[:, 0] + 1) * ((c - c_mean) / c_std) + s[:, 1]       
        
        dlatents_p = Lambda(lambda x: x[:, layer_idx])(dlatents)
        dlatents_p = EqualizedLRDense(num_chs * 2)(dlatents_p) #?
        x = _gen_first_layer_block(dlatents_p) 
        
        layer_idx +=1
        x = EqualizedLRConv2D(self._cal_num_chs(res - 1), 3, padding='same')(x)
        x = self._gen_final_layer_block(x, dlatents, layer_idx) 
        
        # Middle layers.
        res = 3
        while res <= res_log2:
            layer_idx = res * 2 - 4
            
            if np.min(K.int_shape(x)[2:]) * 2 >= 128: #?    
                x = FusedEqualizedLRConv2DTranspose(filters=self._cal_num_chs(res - 1)
                                    , kernel_size=3 #?
                                    , strides=2
                                    , padding='same')(x)
            else:
                x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
                x = EqualizedLRConv2D(self._cal_num_chs(res - 1), 3, padding='same')(x)
            
            x = BlurDepthwiseConv2D(padding='same')(x) #?     
            x = self._gen_final_layer_block(x, dlatents, layer_idx) 
            
            layer_idx = res * 2 - 3            
            x = EqualizedLRConv2D(self._cal_num_chs(res - 1)
                       , 3
                       , padding='same')(x)
            x = self._gen_final_layer_block(x, dlatents, layer_idx) 
            
            res +=1
        
        # Last layer.
        output1 = EqualizedLRConv2D(3
                        , 1
                        , strides=1
                        , activation='tanh' #?
                        , padding='same')(x)

        if self.nn_arch['label_usage']:
            self.gen = ModelExt(inputs=inputs1 + [inputs2[0]]
                                , outputs=[output1] #, output2]
                                , name='gen')
        else:
            self.gen = ModelExt(inputs=[inputs1, inputs2]
                                , outputs=[output1]
                                , name='gen')

    def _gen_final_layer_block(self, x, dlatents, layer_idx):
        """ Generator's final layer block. 
        
        Parameters
        ----------
        x: Tensor.
            Input tensor.
        dlatents: Tensor.
            Disentangeled latent tensor.
        layer_idx: Integer.
            Layer index.
        
        Returns
        -------
        Final layer's tensor.
            Tensor.
        """
        x = StyleGAN._apply_noise_layer(x)
        x = LeakyReLU(0.2)(x)
        x = Lambda(lambda x: x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + 1e-8))(x)
        dlatents_p = Lambda(lambda x: x[:, layer_idx])(dlatents)
        dlatents_p = EqualizedLRDense(K.int_shape(x)[-1] * 2)(dlatents_p) #?
        x = AdaptiveINWithStyle()([x, dlatents_p])       
        
        return x

    @staticmethod
    def _apply_noise_layer(x):
        n = K.random_normal(K.int_shape(x)[1:])
        w = K.variable(np.ones(shape=K.int_shape(x)[-1]))
        return x + n * K.reshape(w, (1, 1, 1, -1))
                                 
    def _create_mapping_net(self):
        """Create mapping network."""
        assert 'num_layers' in self.nn_arch.keys()
                
        # Design mapping network.
        # Inputs.
        noises = Input(shape=(self.map_nn_arch['latent_dim'], ))
        x = noises
        
        if self.nn_arch['label_usage']:
            labels = Input(shape=(1, ), dtype='float32')
        
            # Label multiplication.
            l = Flatten()(Embedding(self.map_nn_arch['num_classes']
                              , self.map_nn_arch['latent_dim'])(labels)) #?
            x = Concatenate(-1)([x, l])
        
        # Pixel normalization
        x = Lambda(lambda x: x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + 1e-8))(x)
        
        # Mapping layers.
        for layer_idx in range(self.map_nn_arch['num_layers'] - 1):
            output_dim = self.map_nn_arch['dlatent_dim'] \
                if layer_idx == self.map_nn_arch['num_layers'] - 1 \
                else self.map_nn_arch['dense1_dim']
            
            x = LeakyReLU(0.2)(Dense(output_dim)(x))
                
        output_dim = self.map_nn_arch['dlatent_dim']
        x = LeakyReLU(0.2, name='map_output')(Dense(output_dim)(x))
        num_layers = self.nn_arch['num_layers']
        output = Lambda(lambda x: K.repeat(x, num_layers))(x)
         
        self.map = ModelExt(inputs=[noises, labels] if self.nn_arch['label_usage'] else [noises]
                                    , outputs=[output], name='map')

    def _create_discriminator(self):
        """Create the discriminator."""
        res = self.nn_arch['resolution'] #?
        
        # Design the model according to the final image resolution.
        res_log2 = int(np.log2(res))
        assert res == 2 ** res_log2 and res >= 4 #?

        images = Input(shape=(res, res, 3))
        
        if self.nn_arch['label_usage']:
            labels = Input(shape=(1, ), dtype='float32')
        
        # First layer.
        res = res_log2
        x = EqualizedLRConv2D(self._cal_num_chs(res - 1)
                   , 1
                   , padding='same')(images)
        x = LeakyReLU(0.2)(x)
                
        # Middle layers.
        for res in range(res_log2, 2, -1):
            x = EqualizedLRConv2D(self._cal_num_chs(res - 1)
                   , 3
                   , padding='same')(x)
            x = LeakyReLU(0.2)(x)
            
            x = BlurDepthwiseConv2D(padding='same')(x) #?
            if np.min(K.int_shape(x)[2:]) * 2 >= 128: #?  
                x = FusedEqualizedLRConv2D(self._cal_num_chs(res - 2)
                                , 3
                                , padding='same')(x)
            else:
                x = EqualizedLRConv2D(self._cal_num_chs(res - 2)
                   , 3
                   , padding='same')(x) #?
                x = AveragePooling2D()(x)    
            
            x = LeakyReLU(0.2)(x)                   
        
        # Layer for 4*4 size.
        res = 2
        x = MinibatchStddevConcat()(x)
        x = EqualizedLRConv2D(self._cal_num_chs(res - 1) #?
                   , 3
                   , padding='same')(x) #?
        x = LeakyReLU(0.2)(x)
        x = Flatten()(x)
        x = EqualizedLRDense(self._cal_num_chs(res - 2))(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(rate=self.disc_nn_arch['dropout_rate'])(x)
        x = EqualizedLRDense(1)(x)
        
        # Last layer.        
        if self.nn_arch['label_usage']:
            x = Lambda(lambda x: K.sum(x[0] * K.cast(x[1], dtype=np.float32), axis=1, keepdims=True))([x, labels]) #?
            output = Activation('linear')(x)
            #output = Activation('sigmoid')(x)
            self.disc = ModelExt(inputs=[images, labels], outputs=[output], name='disc')
        else:
            x = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(x)
            output = Activation('linear')(x)
            #output = Activation('sigmoid')(x)
            self.disc = ModelExt(inputs=[images], outputs=[output], name='disc')
 
    def gen_disc_ext_data_fun(self, generator, gen_prog_depth=None, disc_prog_depth=None, *args, **kwargs):
        # Create x, x_tilda.
        # x.
        if self.nn_arch['label_usage']:
            x_inputs1, x_inputs2 = next(generator)
            x_inputs_b = [x_inputs1['inputs1'], x_inputs2['inputs2']]  
        else:
            x_inputs1, x_inputs2 = next(generator)
            x_inputs_b = [x_inputs1['inputs1']]
        
        if disc_prog_depth:
            size = K.int_shape(self.disc.layers[disc_prog_depth].input)[1]
            x_inputs_b[0] = np.asarray([cv.resize(img, (size, size)) for img in x_inputs_b[0]])
            if len(x_inputs_b) == 3:
                x_inputs_b[2] = x_inputs_b[0]
            
        num_samples = x_inputs_b[0].shape[0]
        
        x_outputs_b = [np.ones(shape=tuple([num_samples] + list(self.disc.get_output_shape_at(0)[1:])))]
        
        # x_tilda.
        if self.nn_arch['label_usage']:
            z_inputs_b = [np.random.normal(size=(num_samples, self.map_nn_arch['latent_dim']))] \
                + [np.random.randint(self.map_nn_arch['num_classes'], size=(num_samples, 1))] \
                + [np.random.normal(size=(num_samples, self.map_nn_arch['latent_dim']))]
        else:
            z_inputs_b = [np.random.normal(size=(num_samples, self.map_nn_arch['latent_dim']))] \
                + [np.random.normal(size=(num_samples, self.map_nn_arch['latent_dim']))]
            
        z_outputs_b = [np.zeros(shape=tuple([num_samples] + list(self.disc.get_output_shape_at(0)[1:])))]
        
        #def get_x3_inputs(x):
        #    n = K.random_normal((1, 1, 1))
        #    return [n * x[1] + (1. - n) * x[0]]
        
        #z_outputs = [self.gen(z_inputs_b)] if len(self.gen.outputs) == 1 else self.gen(z_inputs_b)                
        #x3_inputs_b = get_x3_inputs([x_inputs_b[0], z_outputs[0]]) 
        
        return x_inputs_b + z_inputs_b, x_outputs_b + x_outputs_b + z_outputs_b  #?
        #return x_inputs_b + z_inputs_b, x_outputs_b + z_outputs_b  #?         
        
    def gen_gen_disc_data_fun(self, generator, gen_prog_depth=None, disc_prog_depth=None, *args, **kwargs):
        # Create x, x_tilda.
        # x.
        if self.nn_arch['label_usage']:
            x_inputs1, x_inputs2 = next(generator)
            x_inputs_b = [x_inputs1['inputs1'], x_inputs2['inputs2']]  
        else:
            x_inputs1, x_inputs2 = next(generator)
            x_inputs_b = [x_inputs1['inputs1']]
        
        if disc_prog_depth:
            size = K.int_shape(self.disc.layers[disc_prog_depth].input)[1]
            x_inputs_b[0] = np.asarray([cv.resize(img, (size, size)) for img in x_inputs_b[0]])
            if len(x_inputs_b) == 3:
                x_inputs_b[2] = x_inputs_b[0]
            
        num_samples = x_inputs_b[0].shape[0]
        
        x_outputs_b = [np.ones(shape=tuple([num_samples] + list(self.disc.get_output_shape_at(0)[1:])))]
        
        # x_tilda.
        if self.nn_arch['label_usage']:
            z_inputs_b = [np.random.normal(size=(num_samples, self.map_nn_arch['latent_dim']))] \
                + [np.random.randint(self.map_nn_arch['num_classes'], size=(num_samples, 1))] \
                + [np.random.normal(size=(num_samples, self.map_nn_arch['latent_dim']))]
        else:
            z_inputs_b = [np.random.normal(size=(num_samples, self.map_nn_arch['latent_dim']))] \
                + [np.random.normal(size=(num_samples, self.map_nn_arch['latent_dim']))]
            
        z_outputs_b = [np.ones(shape=tuple([num_samples] + list(self.disc.get_output_shape_at(0)[1:])))]

        return z_inputs_b, z_outputs_b
         
    def train(self):
        """Train."""
        
        # Load training data.
        '''
        generator = self.TrainingSequenceUCCS(self.raw_data_path
                                              , self.hps
                                              , self.nn_arch['resolution']
                                              , batch_shuffle=True)
        '''
        
        generator_tr = self.TrainingSequenceFFHQ(self.raw_data_path
                                              , self.hps
                                              , self.nn_arch
                                              , MODE_TRAINING
                                              , val_ratio=0.1
                                              , batch_shuffle=False)
        generator_val = self.TrainingSequenceFFHQ(self.raw_data_path
                                              , self.hps
                                              , self.nn_arch
                                              , MODE_VALIDATION
                                              , val_ratio=0.1
                                              , batch_shuffle=False)        
        
        # Train.
        self.fit_generator(generator_tr
                        , self.gen_disc_ext_data_fun
                        , self.gen_gen_disc_data_fun
                        , validation_data_gen=None #generator_val
                        , max_queue_size=128
                        , workers=4
                        , use_multiprocessing=False
                        , shuffle=True) #?
        '''
        
        self.fit_generator_progressively(generator_tr
                        , self.gen_disc_ext_data_fun
                        , self.gen_gen_disc_data_fun
                        , validation_data_gen=generator_val
                        , max_queue_size=128
                        , workers=8
                        , use_multiprocessing=False
                        , shuffle=True) #?
        '''

    def fit_generator(self
                      , generator
                      , gen_disc_ext_data_fun
                      , gen_gen_disc_data_fun
                      , verbose=1
                      , callbacks_disc_ext_raw=None
                      , callbacks_gen_disc_raw=None
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

            initial_epoch = self.disc_ext._maybe_load_initial_epoch_from_ckpt(initial_epoch) #?
            
            pre_e_i = -1
            for e_i in range(initial_epoch, self.hps['epochs']):
                if callbacks_disc_ext.model.stop_training or callbacks_gen_disc.model.stop_training:
                    break
                
                self.disc_ext.reset_metrics() #?
                self.gen_disc.reset_metrics() 
                    
                epochs_log_disc_ext = {}
                epochs_log_gen_disc = {}
                                                   
                callbacks_disc_ext.on_epoch_begin(e_i, epochs_log_disc_ext)
                callbacks_gen_disc.on_epoch_begin(e_i, epochs_log_gen_disc)

                for s_i in range(self.hps['batch_step']):
                    for k_i in range(self.hps['disc_k_step']):
                        step = self.hps['disc_k_step'] * s_i + k_i + 1 #?
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
                                    , reset_metrics=False) #?
                            del inputs, outputs

                            end_step = step + 1
                            callbacks_disc_ext.on_train_batch_end(end_step, disc_ext_step_logs)

                    step = s_i + 1 #?
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
                                                            , reset_metrics=False)
                        del inputs, outputs

                        end_step = step + 1
                        callbacks_gen_disc.on_train_batch_end(end_step, gen_disc_step_logs)
                    
                disc_ext_epoch_logs = copy.copy(disc_ext_step_logs) #?
                gen_disc_epoch_logs = copy.copy(gen_disc_step_logs)  # ?

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
                        # Make epochs logs.
                        epochs_log_disc_ext.update(val_outs_disc_ext)
                        epochs_log_gen_disc.update(val_outs_gen_disc)
                                                                
                callbacks_disc_ext.on_epoch_end(e_i, disc_ext_epoch_log)
                callbacks_gen_disc.on_epoch_end(e_i, gen_disc_epoch_log)
                disc_ext_training_logs = disc_ext_epoch_logs
                gen_disc_training_logs = gen_disc_epoch_logs

                if save_f:
                    self.save_gan_model()
                
                # Save sample images. 
                input1 = np.random.normal(size=(1, self.map_nn_arch['latent_dim']))
                input2 = np.asarray([1024])
                input3 = np.random.normal(size=(1, self.map_nn_arch['latent_dim']))
                inputs = [input1, input2, input3]
                
                res = self.generate(inputs)
                sample = res[0]
                sample = np.squeeze(sample) * 255.
                sample = sample.astype('uint8')
                imsave(os.path.join('results', 'sample_' + str(e_i) + '.png'), sample, check_contrast=False)
                
                pre_e_i = e_i

            callbacks_disc_ext.on_train_end(logs=disc_ext_training_logs) # progress bar?
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
                initial_epoch)  #?
            
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
                        step = self.hps['disc_k_step'] * s_i + k_i + 1  # ?
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
                                                                              , reset_metrics=False)  # ?
                            del inputs, outputs

                            end_step = step + 1
                            callbacks_disc_ext.on_train_batch_end(end_step, disc_ext_step_logs)

                    step = s_i + 1  # ?
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
                                                                          , reset_metrics=False)
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
                                                                    , callbacks=None  # callbacks_disc_ext
                                                                    , workers=0
                                                                    , verbose=0)

                        # gen_disc.
                        val_outs_gen_disc = self._evaluate_gen_disc(self.gen_disc
                                                                    , val_generator
                                                                    , gen_gen_disc_data_fun
                                                                    , callbacks=None  # callbacks_gen_disc
                                                                    , workers=0
                                                                    , verbose=0)
                        # Make epochs logs.
                        epochs_log_disc_ext.update(val_outs_disc_ext)
                        epochs_log_gen_disc.update(val_outs_gen_disc)

                callbacks_disc_ext.on_epoch_end(e_i, disc_ext_epoch_log)
                callbacks_gen_disc.on_epoch_end(e_i, gen_disc_epoch_log)
                disc_ext_training_logs = disc_ext_epoch_logs
                gen_disc_training_logs = gen_disc_epoch_logs

                if save_f:
                    self.save_gan_model()

                # Save sample images.
                input1 = np.random.normal(size=(1, self.map_nn_arch['latent_dim']))
                input2 = np.asarray([1024])
                input3 = np.random.normal(size=(1, self.map_nn_arch['latent_dim']))
                inputs = [input1, input2, input3]

                res = self.generate(inputs)
                sample = res[0]
                sample = np.squeeze(sample) * 255.
                sample = sample.astype('uint8')
                imsave(os.path.join('results', 'sample_' + str(e_i) + '.png'), sample, check_contrast=False)

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

        for k_i in range(self.hps['batch_step']):
            with trace.Trace('TraceContext', graph_type='test', step_num=(k_i * self.hps['batch_step'] + 1)):
                callbacks.on_test_batch_begin(k_i * self.hps['batch_step' + 1])

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
            with trace.Trace('TraceContext', graph_type='test', step_num=(s_i + 1)):
                callbacks.on_test_batch_begin(s_i + 1)

                inputs, outputs = gen_gen_disc_data_func(generator)
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

    def evaluate(self):
        """Evaluate."""
        
        # Initialize the results directory
        if not os.path.isdir(os.path.join('results')):
            os.mkdir(os.path.join('results'))
        else:
            shutil.rmtree(os.path.join('results'))
            os.mkdir(os.path.join('results'))
        
        # Save sample images. 
        for label_index in range(self.map_nn_arch['num_classes']):
            print(label_index)
            input1 = np.random.normal(size=(1, self.map_nn_arch['latent_dim']))
            input2 = np.asarray([label_index])
            input3 = np.random.normal(size=(1, self.map_nn_arch['latent_dim']))
            inputs = [input1, input2, input3]
            
            res = self.generate(inputs)
            sample = res[0]
            sample = np.squeeze(sample) * 255.
            sample = sample.astype('uint8')
            imsave(os.path.join('results', 'sample_' + str(label_index) + '.png'), sample, check_contrast=False)        
        
    def generate(self, inputs, *args, **kwargs):
        """Generate styled images.
        
        Parameters
        ----------
        inputs: List or tuple.
            Input list.
        """ 
        s_images = super(StyleGAN, self).generate(inputs, *args, **kwargs) #?
        s_images[0] = (s_images[0] * 0.5 + 0.5) #Label?
        return s_images

    class TrainingSequenceFFHQ(Sequence):
        """Training data set sequence for Flickr-Faces-HQ."""
        
        def __init__(self, raw_data_path, hps, nn_arch, mode=MODE_TRAINING, val_ratio=0.1, batch_shuffle=True):
            """
            Parameters
            ----------
            raw_data_path: String.
                Raw data path.
            hps: Dict.
                Hyper-parameters.
            nn_arch: Dict.
                NN arch.
            batch_shuffle:
                Batch shuffling flag.
            """
            self.raw_data_path = raw_data_path
            self.hps = hps
            self.nn_arch = nn_arch
            self.res = self.nn_arch['resolution']
            self.mode = mode
            self.val_ratio = val_ratio
            self.batch_shuffle = batch_shuffle
            
            if self.mode == MODE_TRAINING:
                self.sample_paths = glob.glob(os.path.join(self.raw_data_path, '**/*.png'), recursive=True)
                self.sample_paths = self.sample_paths[:int(len(self.sample_paths) * (1.0 - self.val_ratio))]

                self.total_samples = len(self.sample_paths)
                
                self.batch_size = self.hps['batch_size']
                self.hps['tr_step'] = self.total_samples // self.batch_size
                
                if self.total_samples % self.batch_size != 0:
                    self.temp_step = self.hps['tr_step'] + 1
                else:
                    self.temp_step = self.hps['tr_step']
            elif self.mode == MODE_VALIDATION:
                self.sample_paths = glob.glob(os.path.join(self.raw_data_path, '**/*.png'), recursive=True)
                self.sample_paths = self.sample_paths[int(len(self.sample_paths) * (1.0 - self.val_ratio)):]
                
                self.total_samples = len(self.sample_paths)
            
                self.batch_size = self.hps['batch_size']
                self.hps['val_step'] = self.total_samples // self.batch_size
                
                if self.total_samples % self.batch_size != 0:
                    self.temp_step = self.hps['val_step'] + 1
                else:
                    self.temp_step = self.hps['val_step']
            else:
                raise ValueError('Mode must be assigned.')
                
        def __len__(self):
            return self.temp_step
        
        def __getitem__(self, index):
            images = []
            labels = []
            
            if self.batch_shuffle:
                idxes = np.random.choice(self.total_samples, size=self.batch_size)
                for bi in idxes:
                    image = imread(self.sample_paths[bi])                    
                    image = 2.0 * (image / 255. - 0.5)
                    image = resize(image, (self.res, self.res))
                    
                    images.append(image)
                    
                    if platform.system() == 'Windows':
                        labels.append(int(self.sample_paths[bi].split('\\')[-1].split('.')[0]))
                    else:
                        labels.append(int(self.sample_paths[bi].split('/')[-1].split('.')[0]))                
            else:    
                step = self.temp_step
                                
                if index == (step - 1):
                    for bi in range(index * self.batch_size, self.total_samples):
                        image = imread(self.sample_paths[bi])                    
                        image = 2.0 * (image / 255. - 0.5)
                        image = resize(image, (self.res, self.res))
                        
                        images.append(image)
                        
                        if platform.system() == 'Windows':
                            labels.append(int(self.sample_paths[bi].split('\\')[-1].split('.')[0]))
                        else:
                            labels.append(int(self.sample_paths[bi].split('/')[-1].split('.')[0]))
                else:
                    for bi in range(index * self.batch_size, (index + 1) * self.batch_size):
                        image = imread(self.sample_paths[bi])                    
                        image = 2.0 * (image / 255. - 0.5)
                        image = resize(image, (self.res, self.res))
                        
                        images.append(image)
                        
                        if platform.system() == 'Windows':
                            labels.append(int(self.sample_paths[bi].split('\\')[-1].split('.')[0]))
                        else:
                            labels.append(int(self.sample_paths[bi].split('/')[-1].split('.')[0]))
                
                images = np.asarray(images)
                labels = np.asarray(labels, dtype='float32')
                labels[(self.nn_arch['num_classes'] - 1) < labels] = self.nn_arch['num_classes'] - 1.
                labels[labels < 0] = 0          
                                                                                                                     
            return ({'inputs1': images}
                     , {'inputs2': labels}) 

    class TrainingSequenceUCCS(Sequence):
        """Training data set sequence for UCCS."""
        
        def __init__(self, raw_data_path, hps, nn_arch, batch_shuffle=True):
            """
            Parameters
            ----------
            raw_data_path: String.
                Raw data path.
            hps: Dict.
                Hyper-parameters.
            nn_arch: Dict.
                NN arch.
            batch_shuffle: Boolean.
                Batch shuffling flag.
            """
            
            # Create indexing data of positive and negative cases.
            self.raw_data_path = raw_data_path
            self.hps = hps
            self.nn_arch = nn_arch
            self.res = self.nn_arch['resolution']
            self.batch_shuffle = batch_shuffle
            self.db = pd.read_csv(os.path.join(self.raw_data_path, 'subject_image_db.csv'))
            self.db = self.db.iloc[:, 1:]
            self.total_samples = self.db.shape[0]
            
            self.batch_size = self.hps['batch_size']
            self.hps['step'] = self.total_samples // self.batch_size
            
            if self.total_samples % self.batch_size != 0:
                self.temp_step = self.hps['step'] + 1
            else:
                self.temp_step = self.hps['step']
                
        def __len__(self):
            return self.temp_step
        
        def __getitem__(self, index):
            images = []
            labels = []
            
            if self.batch_shuffle:
                idxes = np.random.choice(self.total_samples, size=self.batch_size)
                for bi in idxes:
                    image = imread(os.path.join(self.raw_data_path
                                                     , 'subject_faces'
                                                     , self.db.loc[bi, 'face_file']))                    
                    image = 2.0 * (image / 255. - 0.5)
                    image = resize(image, self.res)
                    
                    images.append(image)
                    labels.append(self.db.loc[bi, 'subject_id'])                
            else:    
                # Check the last index.
                if index == (self.temp_step - 1):
                    for bi in range(index * self.batch_size, self.total_samples):
                        image = imread(os.path.join(self.raw_data_path
                                                         , 'subject_faces'
                                                         , self.db.loc[bi, 'face_file']))
                        image = 2.0 * (image / 255. - 0.5)
                        image = resize(image, (self.res, self.res))
                        
                        images.append(image)
                        labels.append(self.db.loc[bi, 'subject_id'])
                else:
                    for bi in range(index * self.batch_size, (index + 1) * self.batch_size):
                        image = imread(os.path.join(self.raw_data_path
                                                         , 'subject_faces'
                                                         , self.db.loc[bi, 'face_file']))
                        image = 2.0 * (image / 255. - 0.5)
                        image = resize(image, (self.res, self.res))
                        
                        images.append(image)
                        labels.append(self.db.loc[bi, 'subject_id'])               
                                                                                                                     
            return ({'inputs1': np.asarray(images)}
                     , {'inputs2': np.asarray(labels, dtype=np.int32)}) 
                             
def main():
    """Main."""
    
    # Load configuration.
    with open(os.path.join("style_based_gan_conf.json"), 'r') as f:
        conf = json.load(f)   

    if conf['mode'] == 'train':      
        # Train.
        s_gan = StyleGAN(conf)
        
        ts = time.time()
        s_gan.train()
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
    if conf['mode'] == 'evaluate':
        # Evaluate.
        s_gan = StyleGAN(conf)
        
        ts = time.time()
        s_gan.evaluate()
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))        
               
if __name__ == '__main__':    
    main()               