"""
Created on 2019. 6. 19.

@author: Inwoo Chung (gutomitai@gmail.com)
License: BSD 3 clause.

Revision:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import argparse
import time
import pickle
import platform
import shutil
from random import shuffle
import json

import numpy as np
import pandas as pd
from skimage.io import imread, imsave
from scipy.linalg import norm
import h5py

from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Embedding, Flatten, multiply, LeakyReLU, Conv2D, Conv2DTranspose
from keras.activations import sigmoid
from keras.utils import multi_gpu_model
from keras import optimizers
import keras.backend as K 
from keras.engine.input_layer import InputLayer
from keras.utils import Sequence

from ku.backprop import AbstractGAN
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

    class TrainingSequenceUCCS(Sequence):
        """Training data set sequence."""
        
        def __init__(self, raw_data_path, hps, batch_shuffle=True):
            """
            Parameters
            ----------
            raw_data_path: string
                Raw data path.
            hps: dict
                Hyper-parameters.
            batch_shuffle:
                Batch shuffling flag.
            """
            
            # Create indexing data of positive and negative cases.
            self.raw_data_path = raw_data_path
            self.hps = hps
            self.batch_shuffle = batch_shuffle
            self.db = pd.read_csv(os.path.join(self.raw_data_path, 'subject_image_db.csv'))
            self.db = self.db.iloc[:, 1:]
            self.total_samples = self.db.shape[0]
            
            self.batch_size = self.hps['batch_size']
            self.hps['step'] = self.total_samples // self.batch_size
            
            if self.total_samples % self.batch_size != 0:
                self.hps['temp_step'] = self.hps['step'] + 1
            else:
                self.hps['temp_step'] = self.hps['step']
                
        def __len__(self):
            return self.hps['temp_step']
        
        def __getitem__(self, index):
            images = []
            labels = []
            
            if self.batch_shuffle:
                idxes = np.random.rand(0, self.total_samples, self.batch_size)
                for bi in idxes:
                    image = imread(os.path.join(self.raw_data_path
                                                     , 'subject_faces'
                                                     , self.db.loc[bi, 'face_file']))                    
                    images.append(image/255)
                    labels.append(self.db.loc[bi, 'subject_id'])                
            else:    
                # Check the last index.
                if index == (self.hps['temp_step'] - 1):
                    for bi in range(index * self.batch_size, self.total_samples):
                        image = imread(os.path.join(self.raw_data_path
                                                         , 'subject_faces'
                                                         , self.db.loc[bi, 'face_file']))                    
                        images.append(image/255)
                        labels.append(self.db.loc[bi, 'subject_id'])
                else:
                    for bi in range(index * self.batch_size, (index + 1) * self.batch_size):
                        image = imread(os.path.join(self.raw_data_path
                                                         , 'subject_faces'
                                                         , self.db.loc[bi, 'face_file']))                    
                        images.append(image/255)
                        labels.append(self.db.loc[bi, 'subject_id'])               
                                                                                                                     
            return ({'inputs': np.asarray(images)}
                     , {'outputs': np.asarray(labels, dtype=np.int32)}) 
    
    def __init__(self, conf):
        """
        Parameters
        ----------
        conf: dict
            Configuration.
        """
        (AbstractGAN, super).__init__(conf)
        self.map_hps = conf['map_hps']
        self.map_nn_arch = conf['map_nn_arch']
        self.syn_hps = conf['syn_hps']
        self.syn_nn_arch = conf['syn_nn_arch']
        
        self.dist_hps = conf['dist_hps']
        self.dist_nn_arch = conf['dist_nn_arch']
        
        # Create models.
        if self.conf['model_loading'] != True:
            self._create_generator()
            self._create_discriminator()
        
    def _cal_num_chs(self, layer_idx):
        """Calculate the number of channels for each synthesis layer.
        
        Parameters
        ----------
        layer_idx: integer
            Layer index.
        
        Returns:
            Number of channels for each layer.
                integer
        """
        return np.min(int(self.syn_hps['ch_base']) / (2.0 ** layer_idx), self.syn_hps['max_ch'])
        
    def _create_generator(self):
        """Create generator."""
        # Design generator.
        # Mapping network and synthesis layer.
        self._create_mapping_net()
        self._create_synthesizer()
        
        # Inputs.
        inputs1 = self.map_model.inputs
        output2 = Lambda(lambda x: x)(inputs1[1]) 
        
        # Disentangled latent.
        dlatents = self.map(inputs1)
        dlatents1 = self.syn(dlatents)
    
        # Style mixing regularization.
        inputs2 = [K.random_normal(K.shape(inputs1[0])), inputs1[1]]
        dlatents2 = self.map_model(inputs2)
        dlatents = StyleMixingRegularization(mixing_prob=self.hps['mixing_prob'])([dlatents1, dlatents2])
        
        # Truncation trick.
        output = TruncationTrick(psi = self.hps['trunc_psi']
                 , cutoff = self.hps['trunc_cutoff']
                 , momentum= self.hps['trunc_momentum'])(dlatents)
        
        self.gen = Model(inputs=inputs1, outputs=[output, output2], name='gen')

    def _create_synthesizer(self): #?
        """Create synthesis model."""
        
        # Check exception.
        if hasattr(self, 'mapping_model') != True:
            raise RuntimeError('Mapping model must be created before.')
        
        # Design the model according to the final image resolution.
        res_log2 = int(np.log2(self.syn_nn_arch['resolution']))
        assert self.syn_nn_arch['resolution'] == 2 ** res_log2 and self.syn_nn_arch['resolution'] >= 4 #?
        self.syn_nn_arch['num_layer'] = res_log2 * 2 - 2
        
        # Disentangled latent inputs.
        dlatents = Input(shape=(self.syn_nn_arch['num_layers'], self.map_nn_arch['dlatent_dim']))
        
        # The first constant input layer.
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
        
        # Middle layers.
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
        
        # Last layer.
        output = Conv2D(filters=3
                        , kernel_size=1
                        , strides=1
                        , padding='same')(x)
                        
        self.syn = Model(inputs=[dlatents], outputs=[output], name='syn')
                         
    def _create_mapping_net(self):
        """Create mapping network."""
        
        # Design mapping network.
        # Inputs.
        noises = Input(shape=(self.map_nn_arch['latent_dim'], ))
        x = noises
        
        if self.nn_arch['label_usage']:
            labels = Input(shape=(1, ), dtype=np.int32)
        
            # Label multiplication.
            l = Flatten()(Embedding(self.map_nn_arch['num_classes']
                              , self.map_nn_arch['latent_dim'])(labels))
        
            # L2 normalization.
            x = K.l2_normalize(multiply([x, l]), axis=-1) #?
        
        # Mapping layers.
        for layer_idx in range(self.map_nn_arch['num_layers'] - 1):
            output_dim = self.map_nn_arch['dlatent_dim'] \
                if layer_idx == self.map_nn_arch['num_layers'] - 1 \
                else self.map_nn_arch['desne1_dim']
            
            x = LeakyReLU()(Dense(output_dim)(x))
        
        layer_idx = self.map_nn_arch['num_layers'] - 1
        
        output_dim = self.map_nn_arch['dlatent_dim'] \
            if layer_idx == self.map_nn_arch['num_layers'] - 1 \
            else self.map_nn_arch['desne1_dim']
        
        output = LeakyReLU(name='map_output')(Dense(output_dim)(x)) 
        
        self.map = Model(inputs=[noises, labels] if self.nn_arch['label_usage'] else [noises]
                                    , outputs=[output], name='map')

    def _create_discriminator(self):
        """Create the discriminator."""
        res = self.dist_nn_arch['resolution']
        
        # Design the model according to the final image resolution.
        res_log2 = int(np.log2(res))
        assert res == 2 ** res_log2 and res >= 4 #?

        images = Input(shape=(res, res, 3))
        
        if self.nn_arch['label_usage']:
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
        if self.nn_arch['label_usage']:
            output = sigmoid(K.sum(x * labels, axis=1, keepdims=True)) #?
            self.dist = Model(inputs=[images, labels], outputs=[output], name='dist')
        else:
            output = sigmoid(K.sum(x, axis=1, keepdims=True)) #?
            self.dist = Model(inputs=[images], outputs=[output], name='dist')
            
    def train(self):
        """Train."""
        
        # Load training data.
        generator = self.TrainingSequenceUCCS(self.raw_data_path
                                              , self.hps
                                              , batch_shuffle=True)
        
        # Train.
        self.fit_generator(generator
                           , max_queue_size=10
                           , workers=1
                           , use_multiprocessing=True
                           , shuffle=True) #?

    def generate(self, images, labels, *args, **kwargs):
        """Generate styled images.
        
        Parameters
        ----------
        images: 4d numpy array
            Images.
        labels: 2d numpy array
            Labels.
        """ 
        (AbstractGAN, super).generate(self, *args, **kwargs) #?
        
        if self.conf['multi_gpu']:
            if self.nn_arch['label_usage']:
                s_images = self.gen_p.predict([images, labels])
            else:
                s_images = self.gen_p.predict([images])
        else:
            if self.nn_arch['label_usage']:
                s_images = self.gen.predict([images, labels])
            else:
                s_images = self.gen.predict([images])
        
        return s_images
                
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
        
if __name__ == '__main__':    
    main()               