"""
Created on 2019. 6. 19.

@author: Inwoo Chung (gutomitai@gmail.com)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import json
import os

import numpy as np
import pandas as pd

from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Embedding, Flatten, multiply, LeakyReLU, Conv2D, Conv2DTranspose
from keras.activations import sigmoid
from keras.utils import multi_gpu_model
from keras import optimizers
import keras.backend as K 
from keras.engine.input_layer import InputLayer

from ku.backprop import AbstractGAN, gan_gen_loss, gan_disc_loss
from ku.layer_ext import AdaptiveIN
from ku.layer_ext.style import TruncationTrick, StyleMixingRegularization

#os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# Constants.
DEBUG = True
MULTI_GPU = False
NUM_GPUS = 4

class StyleGAN(AbstractGAN):
    """Stype based GAN."""
    
    def __init__(self, hps, nn_arch):
        (AbstractGAN, super).__init__(hps, nn_arch) #?
        self.gen_hps = self.hps['gen_hps']
        self.gen_nn_arch = self.hps['gen_nn_arch']
        self.map_hps = self.gen_hps['map_hps']
        self.map_nn_arch = self.gen_nn_arch['map_nn_arch']
        self.syn_hps = self.gen_hps['syn_hps']
        self.syn_nn_arch = self.gen_hps['syn_nn_arch']
        
        self.dist_hps = self.hps['dist_hps']
        self.dist_nn_arch = self.hps['dist_nn_arch']
        
        # Create models.
        self._create_generator()
        self._create_discriminator()
        
    def _cal_num_chs(self, layer_idx):
        """Calculate the number of channels for each synthesis layer."""
        return np.min(int(self.syn_hps['ch_base']) / (2.0 ** layer_idx), self.syn_hps['max_ch'])
        
    def _create_generator(self):
        """Create generator."""
        
        # Design generator.
        # Mapping network and synthesis layer.
        self._create_mapping_net()
        #self._create_synthesis_layer() #?
        
        # Inputs.
        inputs1 = self.map_model.inputs
        output2 = Lambda(lambda x: x)(inputs1[1]) 
        
        # Disentangled latent.
        dlatents = self.map_model(inputs1)
                
        # Synthesis layer.
        res_log2 = int(np.log2(self.syn_nn_arch['resolution']))
        assert self.syn_nn_arch['resolution'] == 2 ** res_log2 and self.syn_nn_arch['resolution'] >= 4 #?
        self.syn_nn_arch['num_layer'] = res_log2 * 2 - 2
                
        layer_idx = 0
        x = K.constant(1.0, shape=tuple([4, 4, self._cal_num_chs(1)])) #?
        n = K.random_normal(K.int_shape(x)) #?
        w = K.variable(np.random.RandomState().randn(K.int_shape(x)[-1]))
        
        x = Lambda(lambda x: x[0] + x[1] * K.reshape(x[2], (1, 1, -1)))([x, n, w]) #?
        x = LeakyReLU()(x)
        x = AdaptiveIN()([x, Lambda(lambda x: x[..., layer_idx])(dlatents)]) # Pixel normalization?
        
        layer_idx +=1
        x = LeakyReLU()(Conv2D(self._cal_num_chs(layer_idx), 3, padding='same')(x))
        x = AdaptiveIN()([x, Lambda(lambda x: x[..., layer_idx])(dlatents)])
        
        while layer_idx <= res_log2:
            x = Conv2DTranspose(filters=self._cal_num_chs(layer_idx) #?
                                , kernel_size=3
                                , strides=2
                                , padding='same')(x) # Blur?
                                
            n = K.random_normal(K.int_shape(x)) #?
            w = K.variable(np.random.RandomState().randn(K.int_shape(x)[-1]))
            
            x = Lambda(lambda x: x[0] + x[1] * K.reshape(x[2], (1, 1, -1)))([x, n, w]) #?
            x = LeakyReLU()(x)
            x = AdaptiveIN()([x, Lambda(lambda x: x[..., (layer_idx + 1)*2 - 4])(dlatents)]) # Pixel normalization?
                        
            x = Conv2D(filters=self._cal_num_chs(layer_idx)
                       , kernel_size=3
                       , strides=1
                       , padding='same')(x)
        
            n = K.random_normal(K.int_shape(x)) #?
            w = K.variable(np.random.RandomState().randn(K.int_shape(x)[-1]))
            
            x = Lambda(lambda x: x[0] + x[1] * K.reshape(x[2], (1, 1, -1)))([x, n, w]) #?
            x = LeakyReLU()(x)
            x = AdaptiveIN()([x, Lambda(lambda x: x[..., (layer_idx + 1)*2 - 3])(dlatents)]) # Pixel normalization?
            
            layer_idx +=1
        
        dlatents1 = Conv2D(filters=3
                        , kernel_size=1
                        , strides=1
                        , padding='same')(x)        

        # Style mixing regularization.
        inputs2 = [K.random_normal(K.shape(inputs1[0])), inputs1[1]]
        dlatents2 = self.map_model(inputs2)
        dlatents = StyleMixingRegularization(mixing_prob=self.gen_hps['mixing_prob'])([dlatents1, dlatents2])
        
        # Truncation trick.
        output = TruncationTrick(psi = self.gen_hps['trunc_psi']
                 , cutoff = self.gen_hps['trunc_cutoff']
                 , momentum= self.gen_hps['trunc_momentum'])(dlatents)
        
        self.gen = Model(inputs=inputs1, outputs=[output, output2])
                         
    def _create_mapping_net(self):
        """Create mapping network."""
        
        # Design mapping network.
        # Inputs.
        input_noise = Input(shape=(self.gen_nn_arch['latent_dim'], ))
        input_label = Input(shape=(1, ), dtype=np.int32)
        
        # Label multiplication.
        l = Flatten()(Embedding(self.gen_nn_arch['num_classes']
                              , self.gen_nn_arch['latent_dim'])(input_label))
        
        # L2 normalization.
        x = K.l2_normalize(multiply([input_noise, l]), axis=-1) #?
        
        # Mapping layers.
        for layer_idx in range(self.gen_nn_arch['num_layers'] - 1):
            output_dim = self.gen_nn_arch['dlatent_dim'] \
                if layer_idx == self.gen_nn_arch['num_layers'] - 1 \
                else self.gen_nn_arch['desne1_dim']
            
            x = LeakyReLU()(Dense(output_dim)(x))
        
        layer_idx = self.gen_nn_arch['num_layers'] - 1
        
        output_dim = self.gen_nn_arch['dlatent_dim'] \
            if layer_idx == self.gen_nn_arch['num_layers'] - 1 \
            else self.gen_nn_arch['desne1_dim']
        
        output = LeakyReLU(name='map_output')(Dense(output_dim)(x)) 
        
        self.map_model = Model(inputs=[input_noise, input_label]
                                    , outputs=[output])

    def _create_discriminator(self):
        """Create the discriminator."""
        res = self.dist_nn_arch['resolution']
        
        # Design the model according to the final image resolution.
        res_log2 = int(np.log2(res))
        assert res == 2 ** res_log2 and res >= 4 #?

        images = Input(shape=(res, res, 3))
        labels = Input(shape=(1, ), dtype=np.int32)
        
        # First layer.
        x = Conv2D(filters=self._cal_num_chs(res_log2 - 1) #?
                   , kernel_size=1
                   , padding='same'
                   , activation=LeakyReLU())(images) #?
                
        # Middle layers.
        for i in range(res_log2, 3, -1):
            x = Conv2D(filters=self._cal_num_chs(i - 1) #?
                   , kernel_size=1
                   , padding='same'
                   , activation=LeakyReLU())(x) #?
            x = Conv2DTranspose(filters=self._cal_num_chs(i - 2)
                                , kernel_size=3
                                , strides=2
                                , padding='same'
                                , activation=LeakyReLU())(x) #?
        
        # Layer for 4*4 size.
        x = Conv2D(filters=self._cal_num_chs(1) #?
                   , kernel_size=3
                   , padding='same'
                   , activation=LeakyReLU())(x) #?
        x = Dense(self._cal_num_chs(0), activation=LeakyReLU())(x)
        x = Dense(1)(x)
        
        # Last layer.
        output = sigmoid(K.sum(x * labels, axis=1, keepdims=True)) #?
        
        self.dist = Model(inputs=[images, labels], outputs=[output])       
                   