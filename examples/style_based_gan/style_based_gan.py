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
import warnings
import glob

import numpy as np
import pandas as pd
from skimage.io import imread, imsave
import cv2 as cv

from keras.models import Model
from keras.layers import Input, Dense, Lambda, Embedding, Flatten, Multiply, multiply
from keras.layers import LeakyReLU, Conv2D, Conv2DTranspose, DepthwiseConv2D, Activation
import keras.backend as K 
from keras.utils import Sequence, GeneratorEnqueuer, OrderedEnqueuer
from keras.engine.training_utils import iter_sequence_infinite
from keras.utils import plot_model
from keras.utils.generic_utils import to_list, CustomObjectScope
from keras import callbacks as cbks, initializers

from ku.backprop import AbstractGAN
from ku.layer_ext import AdaptiveINWithStyle, TruncationTrick, StyleMixingRegularization, InputVariable
from numpy.linalg.linalg import norm

#os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# Constants.
DEBUG = True
MULTI_GPU = False
NUM_GPUS = 4

def resize_image(image, res):
    """Resize an image according to resolution.
    
    Parameters
    ----------
    image: 3d numpy array
        Image data.
    res: Integer
        Symmetric image resolution.
    
    Returns
    -------
    3d numpy array
        Resized image data.
    """
    
    # Adjust the original image size into the normalized image size according to the ratio of width, height.
    w = image.shape[1]
    h = image.shape[0]
    pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0
                    
    if w >= h:
        w_p = res
        h_p = int(h / w * res)
        pad = res - h_p
        
        if pad % 2 == 0:
            pad_t = pad // 2
            pad_b = pad // 2
        else:
            pad_t = pad // 2
            pad_b = pad // 2 + 1

        image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
        image = cv.copyMakeBorder(image, pad_t, pad_b, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0]) # 416x416?  
    else:
        h_p = res
        w_p = int(w / h * res)
        pad = res - w_p
        
        if pad % 2 == 0:
            pad_l = pad // 2
            pad_r = pad // 2
        else:
            pad_l = pad // 2
            pad_r = pad // 2 + 1                
        
        image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
        image = cv.copyMakeBorder(image, 0, 0, pad_l, pad_r, cv.BORDER_CONSTANT, value=[0, 0, 0]) # 416x416? 
    
    return image

class StyleGAN(AbstractGAN):
    """Style based GAN."""

    class TrainingSequenceFFHQ(Sequence):
        """Training data set sequence for Flickr-Faces-HQ."""
        
        def __init__(self, raw_data_path, hps, res, batch_shuffle=True):
            """
            Parameters
            ----------
            raw_data_path: string
                Raw data path.
            hps: dict
                Hyper-parameters.
            res: Integer
                Symmetric image resolution.
            batch_shuffle:
                Batch shuffling flag.
            """
            self.raw_data_path = raw_data_path
            self.hps = hps
            self.res = res
            self.batch_shuffle = batch_shuffle
            
            self.sample_paths = glob.glob(os.path.join(self.raw_data_path, '**/*.png'), recursive=True)
            self.total_samples = len(self.sample_paths)
            
            self.batch_size = self.hps['mini_batch_size']
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
                idxes = np.random.choice(self.total_samples, size=self.batch_size)
                for bi in idxes:
                    image = imread(self.sample_paths[bi])                    
                    #image = image/255
                    image = resize_image(image, self.res)
                    
                    images.append(image)
                    
                    if platform.system() == 'Windows':
                        labels.append(int(self.sample_paths[bi].split('\\')[-1].split('.')[0]))
                    else:
                        labels.append(int(self.sample_paths[bi].split('/')[-1].split('.')[0]))                
            else:    
                # Check the last index.
                if index == (self.hps['temp_step'] - 1):
                    for bi in range(index * self.batch_size, self.total_samples):
                        image = imread(self.sample_paths[bi])                    
                        #image = image/255
                        image = resize_image(image, self.res)
                        
                        images.append(image)
                        
                        if platform.system() == 'Windows':
                            labels.append(int(self.sample_paths[bi].split('\\')[-1].split('.')[0]))
                        else:
                            labels.append(int(self.sample_paths[bi].split('/')[-1].split('.')[0]))
                else:
                    for bi in range(index * self.batch_size, (index + 1) * self.batch_size):
                        image = imread(self.sample_paths[bi])                    
                        #image = image/255
                        image = resize_image(image, self.res)
                        
                        images.append(image)
                        
                        if platform.system() == 'Windows':
                            labels.append(int(self.sample_paths[bi].split('\\')[-1].split('.')[0]))
                        else:
                            labels.append(int(self.sample_paths[bi].split('/')[-1].split('.')[0]))           
                                                                                                                     
            return ({'inputs1': np.asarray(images)}
                     , {'inputs2': np.asarray(labels, dtype=np.int32)}) 

    class TrainingSequenceUCCS(Sequence):
        """Training data set sequence for UCCS."""
        
        def __init__(self, raw_data_path, hps, res, batch_shuffle=True):
            """
            Parameters
            ----------
            raw_data_path: string
                Raw data path.
            hps: dict
                Hyper-parameters.
            res: Integer
                Symmetric image resolution.
            batch_shuffle:
                Batch shuffling flag.
            """
            
            # Create indexing data of positive and negative cases.
            self.raw_data_path = raw_data_path
            self.hps = hps
            self.res = res
            self.batch_shuffle = batch_shuffle
            self.db = pd.read_csv(os.path.join(self.raw_data_path, 'subject_image_db.csv'))
            self.db = self.db.iloc[:, 1:]
            self.total_samples = self.db.shape[0]
            
            self.batch_size = self.hps['mini_batch_size']
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
                idxes = np.random.choice(self.total_samples, size=self.batch_size)
                for bi in idxes:
                    image = imread(os.path.join(self.raw_data_path
                                                     , 'subject_faces'
                                                     , self.db.loc[bi, 'face_file']))                    
                    image = image/255
                    image = resize_image(image, self.res)
                    
                    images.append(image)
                    labels.append(self.db.loc[bi, 'subject_id'])                
            else:    
                # Check the last index.
                if index == (self.hps['temp_step'] - 1):
                    for bi in range(index * self.batch_size, self.total_samples):
                        image = imread(os.path.join(self.raw_data_path
                                                         , 'subject_faces'
                                                         , self.db.loc[bi, 'face_file']))
                        image = image/255
                        image = resize_image(image, self.res)
                        
                        images.append(image)
                        labels.append(self.db.loc[bi, 'subject_id'])
                else:
                    for bi in range(index * self.batch_size, (index + 1) * self.batch_size):
                        image = imread(os.path.join(self.raw_data_path
                                                         , 'subject_faces'
                                                         , self.db.loc[bi, 'face_file']))
                        image = image/255
                        image = resize_image(image, self.res)
                        
                        images.append(image)
                        labels.append(self.db.loc[bi, 'subject_id'])               
                                                                                                                     
            return ({'inputs1': np.asarray(images)}
                     , {'inputs2': np.asarray(labels, dtype=np.int32)}) 
    
    def __init__(self, conf):
        """
        Parameters
        ----------
        conf: dict
            Configuration.
        """
        self.conf = conf #?
        self.raw_data_path = self.conf['raw_data_path']
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']
        
        self.map_hps = conf['map_hps']
        self.map_nn_arch = conf['map_nn_arch']
        self.syn_hps = conf['syn_hps']
        self.syn_nn_arch = conf['syn_nn_arch']
        
        self.disc_hps = conf['disc_hps']
        self.disc_nn_arch = conf['disc_nn_arch']
        
        # Create models.
        if self.conf['model_loading'] != True:
            self._create_generator_2()
            self._create_discriminator()
            self.compile2()
            
        self.custom_objects = {'AdaptiveINWithStyle': AdaptiveINWithStyle
                               , 'TruncationTrick': TruncationTrick
                               , 'StyleMixingRegularization': StyleMixingRegularization
                               , 'InputVariable': InputVariable}
        
        super(StyleGAN, self).__init__(conf) #?
                
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
        return int(np.min([int(self.syn_hps['ch_base']) / (2.0 ** layer_idx)
                           , self.syn_hps['max_ch']]))
        
    def _create_generator(self):
        """Create generator."""
        # Design generator.
        # Mapping network and synthesis layer.
        self._create_synthesizer()
        self._create_mapping_net()
        
        # Inputs.
        inputs1 = self.map.inputs
        
        if self.nn_arch['label_usage']:
            output2 = Lambda(lambda x: x)(inputs1[1]) 
        
        # Disentangled latent.
        dlatents1 = self.map(inputs1)
    
        # Style mixing regularization.
        if self.nn_arch['label_usage']:
            inputs2 = [Input(shape=K.int_shape(inputs1[0])[1:]), inputs1[1]] # Normal random input.
        else:
            inputs2 = Input(shape=K.int_shape(inputs1[0])[1:]) # Normal random input.
        
        dlatents2 = self.map(inputs2)
        dlatents = StyleMixingRegularization(mixing_prob=self.hps['mixing_prob'])([dlatents1, dlatents2])
        
        # Truncation trick.
        dlatents = TruncationTrick(psi=self.hps['trunc_psi']
                 , cutoff=self.hps['trunc_cutoff']
                 , momentum=self.hps['trunc_momentum'])(dlatents)

        output1 = self.syn([dlatents] + self.syn.inputs[1:]) #?
        
        if self.nn_arch['label_usage']:
            self.gen = Model(inputs=inputs1 + [inputs2[0]] + self.syn.inputs[1:], outputs=[output1, output2], name='gen') #?
        else:
            self.gen = Model(inputs=[inputs1, inputs2] + self.syn.inputs[1:], outputs=[output1], name='gen') #?

    def _create_generator_2(self):
        """Create generator."""
        # Design generator.
        # Mapping network and synthesis layer.
        self._create_synthesizer()
        self._create_mapping_net()
        
        # Inputs.
        inputs1 = self.map.inputs
        
        if self.nn_arch['label_usage']:
            output2 = Lambda(lambda x: x)(inputs1[1]) 
        
        # Disentangled latent.
        dlatents1 = self.map(inputs1)
    
        # Style mixing regularization.
        if self.nn_arch['label_usage']:
            inputs2 = [Input(shape=K.int_shape(inputs1[0])[1:]), inputs1[1]] # Normal random input.
        else:
            inputs2 = Input(shape=K.int_shape(inputs1[0])[1:]) # Normal random input.
        
        dlatents2 = self.map(inputs2)
        dlatents = StyleMixingRegularization(mixing_prob=self.hps['mixing_prob'])([dlatents1, dlatents2])
        
        # Truncation trick.
        dlatents = TruncationTrick(psi=self.hps['trunc_psi']
                 , cutoff=self.hps['trunc_cutoff']
                 , momentum=self.hps['trunc_momentum'])(dlatents)

        # Design the model according to the final image resolution.
        res_log2 = int(np.log2(self.syn_nn_arch['resolution']))
        assert self.syn_nn_arch['resolution'] == 2 ** res_log2 and self.syn_nn_arch['resolution'] >= 4 #?
        self.syn_nn_arch['num_layers'] = res_log2 * 2 - 2
        internal_inputs = []
                
        # The first constant input layer.
        res = 2
        layer_idx = 0
        
        # Inputs.
        x = Input(shape=(1,))
        n = Input(shape=tuple([4, 4, self._cal_num_chs(res - 1)])) # Random noise input.
        w = Input(shape=(1,))
        internal_inputs.append(x)
        internal_inputs.append(n)
        internal_inputs.append(w)
        
        # Input variables.
        x = InputVariable(shape=tuple([4, 4, self._cal_num_chs(res - 1)]))(x)        
        w = InputVariable(shape=(K.int_shape(x)[-1], )
                          , variable_initializer=initializers.Ones())(w)
               
        x = Lambda(lambda x: x[0] + x[1] * K.reshape(x[2], (1, 1, 1, -1)))([x, n, w]) # Broadcasting?
        x = LeakyReLU()(x)
        x = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x) # Pixelwise normalization.
        dlatents_p = Lambda(lambda x: x[:, layer_idx])(dlatents)
        dlatents_p = Dense(K.int_shape(x)[-1] * 2)(dlatents_p)
        x = AdaptiveINWithStyle()([x, dlatents_p]) #?
        
        layer_idx +=1
        x = Conv2D(self._cal_num_chs(res - 1), 3, padding='same')(x)
        
        # Inputs.
        n = Input(shape=K.int_shape(x)[1:]) # Random noise input.
        w = Input(shape=(1,))

        internal_inputs.append(n)
        internal_inputs.append(w)
        
        # Input variables.
        w = InputVariable(shape=(K.int_shape(x)[-1], )
                          , variable_initializer=initializers.Ones())(w)
        x = Lambda(lambda x: x[0] + x[1] * K.reshape(x[2], (1, 1, 1, -1)))([x, n, w]) # Broadcasting??
        x = LeakyReLU()(x)
        x = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x) # Pixelwise normalization. 
        dlatents_p = Lambda(lambda x: x[:, layer_idx])(dlatents)
        dlatents_p = Dense(K.int_shape(x)[-1] * 2)(dlatents_p)
        x = AdaptiveINWithStyle()([x, dlatents_p])
        
        # Middle layers.
        res = 3
        while res <= res_log2:
            layer_idx = res * 2 - 3
            x = Conv2DTranspose(filters=self._cal_num_chs(res - 1) #?
                                , kernel_size=3
                                , strides=2
                                , padding='same')(x) # Blur?
                                
            # Inputs.
            n = Input(shape=K.int_shape(x)[1:]) # Random noise input.
            w = Input(shape=(1,))
    
            internal_inputs.append(n)
            internal_inputs.append(w)
            
            # Input variables.
            w = InputVariable(shape=(K.int_shape(x)[-1], )
                              , variable_initializer=initializers.Ones())(w)
            
            x = Lambda(lambda x: x[0] + x[1] * K.reshape(x[2], (1, 1, 1, -1)))([x, n, w]) # Broadcasting??
            x = LeakyReLU()(x)
            x = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x) # Pixelwise normalization.
            dlatents_p = Lambda(lambda x: x[:, layer_idx])(dlatents)
            dlatents_p = Dense(K.int_shape(x)[-1] * 2)(dlatents_p)
            x = AdaptiveINWithStyle()([x, dlatents_p])
            
            layer_idx = res * 2 - 4            
            x = Conv2D(filters=self._cal_num_chs(res - 1)
                       , kernel_size=3
                       , strides=1
                       , padding='same')(x)
        
            # Inputs.
            n = Input(shape=K.int_shape(x)[1:]) # Random noise input.
            w = Input(shape=(1,))
    
            internal_inputs.append(n)
            internal_inputs.append(w)
            
            # Input variables.
            w = InputVariable(shape=(K.int_shape(x)[-1], )
                              , variable_initializer=initializers.Ones())(w) 
            
            x = Lambda(lambda x: x[0] + x[1] * K.reshape(x[2], (1, 1, 1, -1)))([x, n, w]) # Broadcasting??
            x = LeakyReLU()(x)
            x = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x) # Pixelwise normalization.
            dlatents_p = Lambda(lambda x: x[:, layer_idx])(dlatents)
            dlatents_p = Dense(K.int_shape(x)[-1] * 2)(dlatents_p)
            x = AdaptiveINWithStyle()([x, dlatents_p])
            
            res +=1
        
        # Last layer.
        output1 = Conv2D(filters=3
                        , kernel_size=1
                        , strides=1
                        , activation='tanh'
                        , padding='same')(x)
        output1 = Lambda(lambda x: x * 0.5 + 0.5)(output1)

        if self.nn_arch['label_usage']:
            self.gen = Model(inputs=inputs1 + [inputs2[0]] + internal_inputs, outputs=[output1, output2], name='gen') #?
        else:
            self.gen = Model(inputs=[inputs1, inputs2] + internal_inputs, outputs=[output1], name='gen') #?

    def _create_synthesizer(self):
        """Create synthesis model."""
        
        # Check exception.?        
        # Design the model according to the final image resolution.
        res_log2 = int(np.log2(self.syn_nn_arch['resolution']))
        assert self.syn_nn_arch['resolution'] == 2 ** res_log2 and self.syn_nn_arch['resolution'] >= 4 #?
        self.syn_nn_arch['num_layers'] = res_log2 * 2 - 2
        internal_inputs = []
        
        # Disentangled latent inputs.
        dlatents = Input(shape=(self.syn_nn_arch['num_layers'], self.map_nn_arch['dlatent_dim']))
        
        # The first constant input layer.
        res = 2
        layer_idx = 0
        
        # Inputs.
        x = Input(shape=(1,))
        n = Input(shape=tuple([4, 4, self._cal_num_chs(res - 1)])) # Random noise input.
        w = Input(shape=(1,))
        internal_inputs.append(x)
        internal_inputs.append(n)
        internal_inputs.append(w)
        
        # Input variables.
        x = InputVariable(shape=tuple([4, 4, self._cal_num_chs(res - 1)]))(x)        
        w = InputVariable(shape=(K.int_shape(x)[-1], )
                          , variable_initializer=initializers.Ones())(w)
               
        x = Lambda(lambda x: x[0] + x[1] * K.reshape(x[2], (1, 1, 1, -1)))([x, n, w]) # Broadcasting?
        x = LeakyReLU()(x)
        x = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x) # Pixelwise normalization.
        dlatents_p = Lambda(lambda x: x[:, layer_idx])(dlatents)
        dlatents_p = Dense(K.int_shape(x)[-1] * 2)(dlatents_p)
        x = AdaptiveINWithStyle()([x, dlatents_p]) #?
        
        layer_idx +=1
        x = Conv2D(self._cal_num_chs(res - 1), 3, padding='same')(x)
        
        # Inputs.
        n = Input(shape=K.int_shape(x)[1:]) # Random noise input.
        w = Input(shape=(1,))

        internal_inputs.append(n)
        internal_inputs.append(w)
        
        # Input variables.
        w = InputVariable(shape=(K.int_shape(x)[-1], )
                          , variable_initializer=initializers.Ones())(w)
        x = Lambda(lambda x: x[0] + x[1] * K.reshape(x[2], (1, 1, 1, -1)))([x, n, w]) # Broadcasting??
        x = LeakyReLU()(x)
        x = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x) # Pixelwise normalization. 
        dlatents_p = Lambda(lambda x: x[:, layer_idx])(dlatents)
        dlatents_p = Dense(K.int_shape(x)[-1] * 2)(dlatents_p)
        x = AdaptiveINWithStyle()([x, dlatents_p])
        
        # Middle layers.
        res = 3
        while res <= res_log2:
            layer_idx = res * 2 - 3
            x = Conv2DTranspose(filters=self._cal_num_chs(res - 1) #?
                                , kernel_size=3
                                , strides=2
                                , padding='same')(x) # Blur?
                                
            # Inputs.
            n = Input(shape=K.int_shape(x)[1:]) # Random noise input.
            w = Input(shape=(1,))
    
            internal_inputs.append(n)
            internal_inputs.append(w)
            
            # Input variables.
            w = InputVariable(shape=(K.int_shape(x)[-1], )
                              , variable_initializer=initializers.Ones())(w)
            
            x = Lambda(lambda x: x[0] + x[1] * K.reshape(x[2], (1, 1, 1, -1)))([x, n, w]) # Broadcasting??
            x = LeakyReLU()(x)
            x = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x) # Pixelwise normalization.
            dlatents_p = Lambda(lambda x: x[:, layer_idx])(dlatents)
            dlatents_p = Dense(K.int_shape(x)[-1] * 2)(dlatents_p)
            x = AdaptiveINWithStyle()([x, dlatents_p])
            
            layer_idx = res * 2 - 4            
            x = Conv2D(filters=self._cal_num_chs(res - 1)
                       , kernel_size=3
                       , strides=1
                       , padding='same')(x)
        
            # Inputs.
            n = Input(shape=K.int_shape(x)[1:]) # Random noise input.
            w = Input(shape=(1,))
    
            internal_inputs.append(n)
            internal_inputs.append(w)
            
            # Input variables.
            w = InputVariable(shape=(K.int_shape(x)[-1], )
                              , variable_initializer=initializers.Ones())(w) 
            
            x = Lambda(lambda x: x[0] + x[1] * K.reshape(x[2], (1, 1, 1, -1)))([x, n, w]) # Broadcasting??
            x = LeakyReLU()(x)
            x = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x) # Pixelwise normalization.
            dlatents_p = Lambda(lambda x: x[:, layer_idx])(dlatents)
            dlatents_p = Dense(K.int_shape(x)[-1] * 2)(dlatents_p)
            x = AdaptiveINWithStyle()([x, dlatents_p])
            
            res +=1
        
        # Last layer.
        output = Conv2D(filters=3
                        , kernel_size=1
                        , strides=1
                        , activation='tanh'
                        , padding='same')(x)
                        
        self.syn = Model(inputs=[dlatents] + internal_inputs, outputs=[output], name='syn') #?
                         
    def _create_mapping_net(self):
        """Create mapping network."""
        
        # Check exception.
        if hasattr(self, 'syn') != True:
            raise RuntimeError('Synthesizer must be created before.')        
        
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
            x = Multiply()([x, l])
            x = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x)
        
        # Mapping layers.
        for layer_idx in range(self.map_nn_arch['num_layers'] - 1):
            output_dim = self.map_nn_arch['dlatent_dim'] \
                if layer_idx == self.map_nn_arch['num_layers'] - 1 \
                else self.map_nn_arch['dense1_dim']
            
            x = LeakyReLU()(Dense(output_dim)(x))
                
        output_dim = self.map_nn_arch['dlatent_dim']
        x = LeakyReLU(name='map_output')(Dense(output_dim)(x))
        num_layers = self.syn_nn_arch['num_layers']
        output = Lambda(lambda x: K.repeat(x, num_layers))(x)
         
        self.map = Model(inputs=[noises, labels] if self.nn_arch['label_usage'] else [noises]
                                    , outputs=[output], name='map')

    def _create_discriminator(self):
        """Create the discriminator."""
        res = self.syn_nn_arch['resolution'] #?
        
        # Design the model according to the final image resolution.
        res_log2 = int(np.log2(res))
        assert res == 2 ** res_log2 and res >= 4 #?

        images = Input(shape=(res, res, 3))
        
        if self.nn_arch['label_usage']:
            labels = Input(shape=(1, ), dtype=np.int32)
        
        # First layer.
        res = res_log2
        x = Conv2D(filters=self._cal_num_chs(res - 1) #?
                   , kernel_size=1
                   , padding='same')(images) #?
        x = LeakyReLU()(x)
                
        # Middle layers.
        for res in range(res_log2, 2, -1):
            x = Conv2D(filters=self._cal_num_chs(res - 1) #?
                   , kernel_size=1
                   , padding='same')(x) #?
            x = LeakyReLU()(x)
            x = DepthwiseConv2D(kernel_size=3, padding='same')(x)
            x = Conv2D(filters=self._cal_num_chs(res - 2) #?
                   , kernel_size=3
                   , padding='same')(x) #?
            x = DepthwiseConv2D(kernel_size=3
                                , padding='same'
                                , strides=2)(x)
            x = LeakyReLU()(x)                   
        
        # Layer for 4*4 size.
        res = 2
        x = Conv2D(filters=self._cal_num_chs(res - 1) #?
                   , kernel_size=3
                   , padding='same')(x) #?
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(self._cal_num_chs(res - 2))(x)
        x = LeakyReLU()(x)
        x = Dense(1)(x)
        
        # Last layer.        
        if self.nn_arch['label_usage']:
            x = Lambda(lambda x: K.sum(x[0] * K.cast(x[1], dtype=np.float32), axis=1, keepdims=True))([x, labels]) #?
            output = Activation('sigmoid')(x)
            self.disc = Model(inputs=[images, labels], outputs=[output], name='disc')
        else:
            x = Lambda(lambda x: K.sum(x[0], axis=1, keepdims=True))(x)
            output = Activation('sigmoid')(x)
            self.disc = Model(inputs=[images], outputs=[output], name='disc')
         
    def train(self):
        """Train."""
        
        # Load training data.
        '''
        generator = self.TrainingSequenceUCCS(self.raw_data_path
                                              , self.hps
                                              , self.syn_nn_arch['resolution']
                                              , batch_shuffle=True)
        '''
        generator = self.TrainingSequenceFFHQ(self.raw_data_path
                                              , self.hps
                                              , self.syn_nn_arch['resolution']
                                              , batch_shuffle=True)        
        
        # Train.
        self.fit_generator2(generator
                           , max_queue_size=10
                           , workers=1
                           , use_multiprocessing=False
                           , shuffle=True) #?

    def fit(self, x_inputs, x_outputs, callbacks_disc_ext=None, callbacks_gan=None, verbose=1):
        """Train the GAN model.
        
        Parameters
        ----------
        x_inputs: list
            Training data numpy array list.
        x_outputs: list
            Ground truth data numpy array list.
        callbacks_disc_ext: list
            Callback list of disc ext (default=None).
        callbacks_gan: list 
            Callback list of gan (default=None).
        verbose: Integer 
            Verbose mode (default=1).
            
        Returns
        -------
        Training history.
            tuple
        """
        
        # Callbacks.
        # disc ext.
        if self.conf['multi_gpu']:
            callback_metrics_disc_ext = self.disc_ext_p.metrics_names
            self.disc_ext_p.history = cbks.History()
            _callbacks = [cbks.BaseLogger(stateful_metrics=self.disc_ext_p.stateful_metric_names)]
            if verbose:
                _callbacks.append(cbks.ProgbarLogger(count_mode='steps'
                                                     , stateful_metrics=self.disc_ext_p.stateful_metric_names))
            _callbacks += (callbacks_disc_ext or []) + [self.disc_ext_p.history]
            callbacks_disc_ext = cbks.CallbackList(_callbacks)
            
            callbacks_disc_ext.set_model(self.disc_ext_p)
            callbacks_disc_ext.set_params({'epochs': self.hps['epochs']
                                           , 'steps': self.hps['batch_step'] #?
                                           , 'verbose': verbose
                                           , 'metrics': callback_metrics_disc_ext})
        else:
            callback_metrics_disc_ext = self.disc_ext.metrics_names
            self.disc_ext.history = cbks.History()
            _callbacks = [cbks.BaseLogger(stateful_metrics=self.disc_ext.stateful_metric_names)]
            if verbose:
                _callbacks.append(cbks.ProgbarLogger(count_mode='steps'
                                                     , stateful_metrics=self.disc_ext.stateful_metric_names))
            _callbacks += (callbacks_disc_ext or []) + [self.disc_ext.history]
            callbacks_disc_ext = cbks.CallbackList(_callbacks)
            
            callbacks_disc_ext.set_model(self.disc_ext)
            callbacks_disc_ext.set_params({'epochs': self.hps['epochs']
                                           , 'steps': self.hps['batch_step'] #?
                                           , 'verbose': verbose
                                           , 'metrics': callback_metrics_disc_ext})
        
        # gan.
        if self.conf['multi_gpu']:
            callback_metrics_gan = self.gan_p.metrics_names
            self.gan_p.history = cbks.History()
            _callbacks = [cbks.BaseLogger(stateful_metrics=self.gan_p.stateful_metric_names)]
            if verbose:
                _callbacks.append(cbks.ProgbarLogger(count_mode='steps'
                                                     , stateful_metrics=self.gan_p.stateful_metric_names))
            _callbacks += (callbacks_gan or []) + [self.gan_p.history]
            callbacks_gan = cbks.CallbackList(_callbacks)
            
            callbacks_gan.set_model(self.gan_p)
            callbacks_gan.set_params({'epochs': self.hps['epochs']
                                           , 'steps': self.hps['batch_step'] #?
                                           , 'verbose': verbose
                                           , 'metrics': callback_metrics_gan})
        else:
            callback_metrics_gan = self.gan.metrics_names
            self.gan.history = cbks.History()
            _callbacks = [cbks.BaseLogger(stateful_metrics=self.gan.stateful_metric_names)]
            if verbose:
                _callbacks.append(cbks.ProgbarLogger(count_mode='steps'
                                                     , stateful_metrics=self.gan.stateful_metric_names))
            _callbacks += (callbacks_gan or []) + [self.gan.history]
            callbacks_gan = cbks.CallbackList(_callbacks)
            
            callbacks_gan.set_model(self.gan)
            callbacks_gan.set_params({'epochs': self.hps['epochs']
                                           , 'steps': self.hps['batch_step'] #?
                                           , 'verbose': verbose
                                           , 'metrics': callback_metrics_gan})
        
        callbacks_disc_ext._call_begin_hook('train')
        callbacks_gan._call_begin_hook('train')            
        
        callbacks_disc_ext.model.stop_training = False #?
        callbacks_gan.model.stop_training = False #?
        
        num_samples = self.hps['mini_batch_size']
        epochs_log = {}
        for e_i in range(self.hps['epochs']):
            if self.conf['multi_gpu']:
                for m in self.disc_ext_p.stateful_metric_functions: #
                    m.reset_states()
                for m in self.gan_p.stateful_metric_functions: #
                    m.reset_states()
            else:
                for m in self.disc_ext.stateful_metric_functions: #
                    m.reset_states()
                for m in self.gan.stateful_metric_functions: #
                    m.reset_states()
            
            callbacks_disc_ext.on_epoch_begin(e_i)
            callbacks_gan.on_epoch_begin(e_i)
            steps_done = 0
            batch_index = 0
            for s_i in range(self.hps['batch_step']):
                k_steps_done = 0
                k_batch_index = 0
                                
                for k_i in range(self.hps['disc_k_step']):
                    # Build batch logs.
                    k_batch_logs = {'batch': k_batch_index, 'size': self.hps['mini_batch_size']}
                    callbacks_disc_ext.batch_begin(batch_index, k_batch_logs)
                
                    # Create x_inputs_b, z_inputs_b, x_outputs_b, z_p_outputs_b, z_outputs_b.
                    x_inputs1_b = [x_inputs[i][np.random.choice(x_inputs[i].shape[0], num_samples)] \
                                  for i in range(len(x_inputs))] #?
                    x_inputs2_b = [x_outputs[i][np.random.choice(x_outputs[i].shape[0], num_samples)] \
                                   for i in range(len(x_outputs))] #?
                    
                    if self.nn_arch['label_usage']:
                        z_inputs_b = [np.random.rand(num_samples, self.map_nn_arch['latent_dim'])] \
                                + [np.random.randint(self.map_nn_arch['num_classes'], size=(num_samples, 1))]
                    else:
                        z_inputs_b = [np.random.rand(num_samples, self.map_nn_arch['latent_dim'])]
                            
                    x_outputs_b = [np.ones(shape=tuple([num_samples] + list(self.disc.output_shape[1:])))]
                    z_outputs_b = [np.ones(shape=tuple([num_samples] + list(self.disc.output_shape[1:])))]
         
                    # Train disc.
                    if self.conf['multi_gpu']:
                        outs = self.disc_ext_p.train_on_batch(x_inputs1_b + x_inputs2_b + z_inputs_b
                                 , x_outputs_b + z_outputs_b) #?                    
                    else:
                        outs = self.disc_ext.train_on_batch(x_inputs1_b +  x_inputs2_b + z_inputs_b
                                 , x_outputs_b + z_outputs_b) #?
                    
                    outs = to_list(outs)
                    
                    if self.conf['multi_gpu']:
                        for l, o in zip(self.disc_ext_p.metrics_names, outs):
                            k_batch_logs[l] = o
                    else:
                        for l, o in zip(self.disc_ext.metrics_names, outs):
                            k_batch_logs[l] = o
        
                    callbacks_disc_ext._call_batch_hook('train', 'end', k_batch_index, k_batch_logs)
                    
                    k_batch_index +=1
                    k_steps_done +=1
                            
                # Build batch logs.
                batch_logs = {'batch': batch_index, 'size': self.hps['mini_batch_size']}
                callbacks_gan.batch_begin(batch_index, batch_logs)
        
                if self.nn_arch['label_usage']:
                    z_inputs_b = [np.random.rand(num_samples, self.map_nn_arch['latent_dim'])] \
                            + [np.random.randint(self.map_nn_arch['num_classes'], size=(num_samples, 1))]
                else:
                    z_inputs_b = [np.random.rand(num_samples, self.map_nn_arch['latent_dim'])]
                
                z_p_outputs_b = [np.ones(shape=tuple([num_samples] + list(self.disc.output_shape[1:])))]

                # Train gan.
                if self.conf['multi_gpu']:
                    outs = self.gan_p.train_on_batch(z_inputs_b, z_p_outputs_b)
                else:
                    outs = self.gan.train_on_batch(z_inputs_b, z_p_outputs_b)
                    
                outs = to_list(outs)
                
                if self.conf['multi_gpu']:
                    for l, o in zip(self.gan_p.metrics_names, outs):
                        batch_logs[l] = o
                else:
                    for l, o in zip(self.gan.metrics_names, outs):
                        batch_logs[l] = o
    
                callbacks_gan._call_batch_hook('train', 'end', batch_index, batch_logs)
                
                batch_index +=1
                steps_done +=1

                if callbacks_disc_ext.model.stop_training \
                    or callbacks_gan.model.stop_training:
                    break #?
            
            callbacks_disc_ext.on_epoch_end(e_i, epochs_log)
            callbacks_gan.on_epoch_end(e_i, epochs_log)

            if callbacks_disc_ext.model.stop_training \
                    or callbacks_gan.model.stop_training:
                break #?            

        callbacks_disc_ext._call_end_hook('train')
        callbacks_gan.call_end_hook('train')        
            
        # Save models.
        self.disc_ext.save(self.DISC_EXT_PATH)
        self.gan.save(self.GAN_PATH)
        
        if self.conf['multi_gpu']:
            return self.disc_ext_p.history, self.gan_p.history
        else:
            return self.disc_ext.history, self.gan.history 

    def fit_generator(self
                      , generator
                      , max_queue_size=10
                      , workers=1
                      , use_multiprocessing=False #?
                      , shuffle=True
                      , callbacks_disc_ext=None
                      , callbacks_gan=None
                      , verbose=1):
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
        callbacks_disc_ext: list
            Callback list of disc ext (default=None).
        callbacks_gan: list 
            Callback list of gan (default=None).
        verbose: Integer 
            Verbose mode (default=1).
            
        Returns
        -------
        Training history.
            tuple
        """
        
        # Check exception.
        if not isinstance(generator, Sequence) and use_multiprocessing and workers > 1:
            warnings.warn(UserWarning('For multi processing, use the instance of Sequence.'))
        
        try:        
            # Get the output generator.
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
            
            # Train.        
            # Callbacks.
            # disc ext.
            if self.conf['multi_gpu']:
                callback_metrics_disc_ext = self.disc_ext_p.metrics_names
                self.disc_ext_p.history = cbks.History()
                _callbacks = [cbks.BaseLogger(stateful_metrics=['loss', 'disc_loss', 'disc_loss_2'] + self.disc_ext_p.stateful_metric_names)]
                if verbose:
                    _callbacks.append(cbks.ProgbarLogger(count_mode='steps'
                                                         , stateful_metrics=['loss', 'disc_loss', 'disc_loss_2'] + self.disc_ext_p.stateful_metric_names))
                _callbacks += (callbacks_disc_ext or []) + [self.disc_ext_p.history]
                callbacks_disc_ext = cbks.CallbackList(_callbacks)
                
                callbacks_disc_ext.set_model(self.disc_ext_p)
                callbacks_disc_ext.set_params({'epochs': self.hps['epochs']
                                               , 'steps': self.hps['batch_step'] * self.hps['disc_k_step']
                                               , 'verbose': verbose
                                               , 'metrics': callback_metrics_disc_ext})
            else:
                callback_metrics_disc_ext = self.disc_ext.metrics_names
                self.disc_ext.history = cbks.History()
                _callbacks = [cbks.BaseLogger(stateful_metrics=['loss', 'disc_loss', 'disc_loss_2'] + self.disc_ext.stateful_metric_names)]
                if verbose:
                    _callbacks.append(cbks.ProgbarLogger(count_mode='steps'
                                                         , stateful_metrics=['loss', 'disc_loss', 'disc_loss_2'] + self.disc_ext.stateful_metric_names))
                _callbacks += (callbacks_disc_ext or []) + [self.disc_ext.history]
                callbacks_disc_ext = cbks.CallbackList(_callbacks)
                
                callbacks_disc_ext.set_model(self.disc_ext)
                callbacks_disc_ext.set_params({'epochs': self.hps['epochs']
                                               , 'steps': self.hps['batch_step'] * self.hps['disc_k_step'] #?
                                               , 'verbose': verbose
                                               , 'metrics': callback_metrics_disc_ext})
            
            # gan.
            if self.conf['multi_gpu']:
                callback_metrics_gan = self.gan_p.metrics_names
                self.gan_p.history = cbks.History()
                _callbacks = [cbks.BaseLogger(stateful_metrics=['loss'] + self.gan_p.stateful_metric_names)]
                if verbose:
                    _callbacks.append(cbks.ProgbarLogger(count_mode='steps'
                                                         , stateful_metrics=['loss'] + self.gan_p.stateful_metric_names))
                _callbacks += (callbacks_gan or []) + [self.gan_p.history]
                callbacks_gan = cbks.CallbackList(_callbacks)
                
                callbacks_gan.set_model(self.gan_p)
                callbacks_gan.set_params({'epochs': self.hps['epochs']
                                               , 'steps': self.hps['batch_step'] #?
                                               , 'verbose': verbose
                                               , 'metrics': callback_metrics_gan})
            else:
                callback_metrics_gan = self.gan.metrics_names
                self.gan.history = cbks.History()
                _callbacks = [cbks.BaseLogger(stateful_metrics=['loss'] + self.gan.stateful_metric_names)]
                if verbose:
                    _callbacks.append(cbks.ProgbarLogger(count_mode='steps'
                                                         , stateful_metrics=['loss'] + self.gan.stateful_metric_names))
                _callbacks += (callbacks_gan or []) + [self.gan.history]
                callbacks_gan = cbks.CallbackList(_callbacks)
                
                callbacks_gan.set_model(self.gan)
                callbacks_gan.set_params({'epochs': self.hps['epochs']
                                               , 'steps': self.hps['batch_step'] #?
                                               , 'verbose': verbose
                                               , 'metrics': callback_metrics_gan})
            
            callbacks_disc_ext.on_train_begin()
            callbacks_gan.on_train_begin()           
                        
            num_samples = self.hps['mini_batch_size']
            epochs_log = {}
            
            for e_i in range(self.hps['epochs']):
                if self.conf['multi_gpu']:
                    for m in self.disc_ext_p.stateful_metric_functions: #
                        m.reset_states()
                    for m in self.gan_p.stateful_metric_functions: #
                        m.reset_states()
                else:
                    for m in self.disc_ext.stateful_metric_functions: #
                        m.reset_states()
                    for m in self.gan.stateful_metric_functions: #
                        m.reset_states()
                
                callbacks_disc_ext.on_epoch_begin(e_i)
                callbacks_gan.on_epoch_begin(e_i)

                for s_i in range(self.hps['batch_step']):                   
                    for k_i in range(self.hps['disc_k_step']): #?
                        # Build batch logs.
                        k_batch_logs = {'batch': self.hps['batch_step'] * s_i + k_i, 'size': self.hps['mini_batch_size']}
                        callbacks_disc_ext.on_batch_begin(self.hps['batch_step'] * s_i + k_i, k_batch_logs)
                        
                        x_inputs1, x_inputs2 = next(output_generator)
                        
                        # Create x_inputs_b, z_inputs_b, x2_outputs_b, z_p_outputs_b, z_outputs_b.
                        x_inputs1_b = [x_inputs1['inputs1']]
                        x_inputs2_b = [x_inputs2['inputs2']]
                        
                        if self.nn_arch['label_usage']:
                            z_inputs_b = [np.random.rand(num_samples, self.map_nn_arch['latent_dim'])] \
                                + [np.random.randint(self.map_nn_arch['num_classes'], size=(num_samples, 1))]
                        else:
                            z_inputs_b = [np.random.rand(num_samples, self.map_nn_arch['latent_dim'])]
                            
                        x_outputs_b = [np.ones(shape=tuple([num_samples] + list(self.disc.output_shape[1:])))]
                        z_outputs_b = [np.zeros(shape=tuple([num_samples] + list(self.disc.output_shape[1:])))]
             
                        # Train disc.
                        if self.conf['model_loading']:
                            # Create normal random inputs.
                            internal_inputs = []
                            internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                               + list(K.int_shape(self.disc_ext.inputs[4]))[1:])))
                                                        
                            for inp in self.disc_ext.inputs[5:]:
                                if K.ndim(inp) == 4:
                                    internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                               + list(K.int_shape(inp)[1:]))))
                                else:
                                    internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                               + list(K.int_shape(inp)[1:])))) # Trivial.
                            
                            if self.conf['multi_gpu']:
                                outs = self.disc_ext_p.train_on_batch(x_inputs1_b + x_inputs2_b + z_inputs_b + internal_inputs
                                         , x_outputs_b + z_outputs_b) #?                    
                            else:
                                outs = self.disc_ext.train_on_batch(x_inputs1_b +  x_inputs2_b + z_inputs_b + internal_inputs
                                         , x_outputs_b + z_outputs_b) #?                        
                        else:
                            # Create normal random inputs.
                            internal_inputs = []
                            internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                               + list(K.int_shape(self.disc_ext.inputs[4]))[1:])))
                                                        
                            for inp in self.disc_ext.inputs[5:]:
                                if K.ndim(inp) == 4:
                                    internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                               + list(K.int_shape(inp)[1:]))))
                                else:
                                    internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                               + list(K.int_shape(inp)[1:])))) # Trivial.
                            
                            if self.conf['multi_gpu']:
                                outs = self.disc_ext_p.train_on_batch(x_inputs1_b + x_inputs2_b + z_inputs_b + internal_inputs
                                         , x_outputs_b + z_outputs_b) #?                    
                            else:
                                outs = self.disc_ext.train_on_batch(x_inputs1_b +  x_inputs2_b + z_inputs_b + internal_inputs
                                         , x_outputs_b + z_outputs_b) #? 

                        #print(s_i, self.map.get_weights()[0])

                        outs = to_list(outs)
                        
                        if self.conf['multi_gpu']:
                            metric_names = self.disc_ext_p.metrics_names
                            metric_names[-1] = 'disc_loss_2'
                        else:
                            metric_names = self.disc_ext.metrics_names
                            metric_names[-1] = 'disc_loss_2'                            
                            
                        for l, o in zip(metric_names, outs):
                            k_batch_logs[l] = o                        
            
                        ws = self.gen.get_weights()
                        res = []
                        for w in ws:
                            res.append(np.isfinite(w).all())
                        res = np.asarray(res)

                        #print('\n', k_batch_logs)
                        callbacks_disc_ext.on_batch_end(self.hps['batch_step'] * s_i + k_i, k_batch_logs)
                        
                    # Build batch logs.
                    batch_logs = {'batch': s_i, 'size': self.hps['mini_batch_size']}
                    callbacks_gan.on_batch_begin(s_i, batch_logs)

                    if self.nn_arch['label_usage']:
                        z_inputs_b = [np.random.rand(num_samples, self.map_nn_arch['latent_dim'])] \
                                + [np.random.randint(self.map_nn_arch['num_classes'], size=(num_samples, 1))]
                    else:
                        z_inputs_b = [np.random.rand(num_samples, self.map_nn_arch['latent_dim'])]
                    
                    z_p_outputs_b = [np.ones(shape=tuple([num_samples] + list(self.disc.output_shape[1:])))]
                    
                    # Train gan.
                    if self.conf['model_loading']:
                        # Create normal random inputs.
                        internal_inputs = []
                        internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                           + list(K.int_shape(self.disc_ext.inputs[4]))[1:])))
                                                    
                        for inp in self.disc_ext.inputs[5:]:
                            if K.ndim(inp) == 4:
                                internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                           + list(K.int_shape(inp)[1:]))))
                            else:
                                internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                           + list(K.int_shape(inp)[1:])))) # Trivial.
                        
                        if self.conf['multi_gpu']:
                            outs = self.gan_p.train_on_batch(z_inputs_b + internal_inputs, z_p_outputs_b)
                        else:
                            outs = self.gan.train_on_batch(z_inputs_b + internal_inputs, z_p_outputs_b)
                    else:
                        # Create normal random inputs.
                        internal_inputs = []
                        internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                           + list(K.int_shape(self.disc_ext.inputs[4]))[1:])))
                                                    
                        for inp in self.disc_ext.inputs[5:]:
                            if K.ndim(inp) == 4:
                                internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                           + list(K.int_shape(inp)[1:]))))
                            else:
                                internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                           + list(K.int_shape(inp)[1:])))) # Trivial.
                        
                        if self.conf['multi_gpu']:
                            outs = self.gan_p.train_on_batch(z_inputs_b + internal_inputs, z_p_outputs_b)
                        else:
                            outs = self.gan.train_on_batch(z_inputs_b + internal_inputs, z_p_outputs_b)

                    outs = to_list(outs)
                    
                    if self.conf['multi_gpu']:
                        for l, o in zip(self.gan_p.metrics_names, outs):
                            batch_logs[l] = o
                    else:
                        for l, o in zip(self.gan.metrics_names, outs):
                            batch_logs[l] = o
        
                    #print('\n', batch_logs)
                    callbacks_gan.on_batch_end(s_i, batch_logs)
                    
                callbacks_disc_ext.on_epoch_end(e_i, epochs_log)
                callbacks_gan.on_epoch_end(e_i, epochs_log)
        
            callbacks_disc_ext.on_train_end() #?
            callbacks_gan.on_train_end() #?  
        finally:
            try:
                if enq is not None:
                    enq.stop()
            finally:
                pass

        # Save models.
        with CustomObjectScope(self.custom_objects):
            self.disc_ext.save(self.DISC_EXT_PATH)
            self.gan.save(self.GAN_PATH)

        if self.conf['multi_gpu']:
            return self.disc_ext_p.history, self.gan_p.history
        else:
            return self.disc_ext.history, self.gan.history

    def fit_generator2(self
                      , generator
                      , max_queue_size=10
                      , workers=1
                      , use_multiprocessing=False #?
                      , shuffle=True
                      , callbacks_disc=None
                      , callbacks_gan=None
                      , verbose=1):
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
        callbacks_disc: list
            Callback list of disc (default=None).
        callbacks_gan: list 
            Callback list of gan (default=None).
        verbose: Integer 
            Verbose mode (default=1).
            
        Returns
        -------
        Training history.
            tuple
        """
        
        # Check exception.
        if not isinstance(generator, Sequence) and use_multiprocessing and workers > 1:
            warnings.warn(UserWarning('For multi processing, use the instance of Sequence.'))
        
        try:        
            # Get the output generator.
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
            
            # Train.        
            # Callbacks.
            # disc.
            if self.conf['multi_gpu']:
                callback_metrics_disc = self.disc_p.metrics_names
                self.disc_p.history = cbks.History()
                _callbacks = [cbks.BaseLogger(stateful_metrics=['loss'] + ['real_loss', 'fake_loss'] + self.disc_p.stateful_metric_names)]
                if verbose:
                    _callbacks.append(cbks.ProgbarLogger(count_mode='steps'
                                                         , stateful_metrics=['loss'] + ['real_loss', 'fake_loss'] + self.disc_p.stateful_metric_names))
                _callbacks += (callbacks_disc or []) + [self.disc_p.history]
                callbacks_disc = cbks.CallbackList(_callbacks)
                
                callbacks_disc.set_model(self.disc_p)
                callbacks_disc.set_params({'epochs': self.hps['epochs']
                                               , 'steps': self.hps['batch_step'] * self.hps['disc_k_step']
                                               , 'verbose': verbose
                                               , 'metrics': callback_metrics_disc})
            else:
                callback_metrics_disc = self.disc.metrics_names
                self.disc.history = cbks.History()
                _callbacks = [cbks.BaseLogger(stateful_metrics=['loss'] + ['real_loss', 'fake_loss'] + self.disc.stateful_metric_names)]
                if verbose:
                    _callbacks.append(cbks.ProgbarLogger(count_mode='steps'
                                                         , stateful_metrics=['loss'] + ['real_loss', 'fake_loss'] + self.disc.stateful_metric_names))
                _callbacks += (callbacks_disc or []) + [self.disc.history]
                callbacks_disc = cbks.CallbackList(_callbacks)
                
                callbacks_disc.set_model(self.disc)
                callbacks_disc.set_params({'epochs': self.hps['epochs']
                                               , 'steps': self.hps['batch_step'] * self.hps['disc_k_step'] #?
                                               , 'verbose': verbose
                                               , 'metrics': callback_metrics_disc})
            
            # gan.
            if self.conf['multi_gpu']:
                callback_metrics_gan = self.gan_p.metrics_names
                self.gan_p.history = cbks.History()
                _callbacks = [cbks.BaseLogger(stateful_metrics=['loss'] + self.gan_p.stateful_metric_names)]
                if verbose:
                    _callbacks.append(cbks.ProgbarLogger(count_mode='steps'
                                                         , stateful_metrics=['loss'] + self.gan_p.stateful_metric_names))
                _callbacks += (callbacks_gan or []) + [self.gan_p.history]
                callbacks_gan = cbks.CallbackList(_callbacks)
                
                callbacks_gan.set_model(self.gan_p)
                callbacks_gan.set_params({'epochs': self.hps['epochs']
                                               , 'steps': self.hps['batch_step'] #?
                                               , 'verbose': verbose
                                               , 'metrics': callback_metrics_gan})
            else:
                callback_metrics_gan = self.gan.metrics_names
                self.gan.history = cbks.History()
                _callbacks = [cbks.BaseLogger(stateful_metrics=['loss'] + self.gan.stateful_metric_names)]
                if verbose:
                    _callbacks.append(cbks.ProgbarLogger(count_mode='steps'
                                                         , stateful_metrics=['loss'] + self.gan.stateful_metric_names))
                _callbacks += (callbacks_gan or []) + [self.gan.history]
                callbacks_gan = cbks.CallbackList(_callbacks)
                
                callbacks_gan.set_model(self.gan)
                callbacks_gan.set_params({'epochs': self.hps['epochs']
                                               , 'steps': self.hps['batch_step'] #?
                                               , 'verbose': verbose
                                               , 'metrics': callback_metrics_gan})
            
            callbacks_disc.on_train_begin()
            callbacks_gan.on_train_begin()           
                        
            num_samples = self.hps['mini_batch_size']
            epochs_log = {}
            
            for e_i in range(self.hps['epochs']):
                if self.conf['multi_gpu']:
                    for m in self.disc_p.stateful_metric_functions: #
                        m.reset_states()
                    for m in self.gan_p.stateful_metric_functions: #
                        m.reset_states()
                else:
                    for m in self.disc.stateful_metric_functions: #
                        m.reset_states()
                    for m in self.gan.stateful_metric_functions: #
                        m.reset_states()
                
                callbacks_disc.on_epoch_begin(e_i)
                callbacks_gan.on_epoch_begin(e_i)

                for s_i in range(self.hps['batch_step']):                   
                    for k_i in range(self.hps['disc_k_step']): #?
                        # Build batch logs.
                        k_batch_logs = {'batch': self.hps['batch_step'] * s_i + k_i, 'size': self.hps['mini_batch_size']}
                        callbacks_disc.on_batch_begin(self.hps['batch_step'] * s_i + k_i, k_batch_logs)
                        
                        x_inputs1, x_inputs2 = next(output_generator)
                        
                        # Create x_inputs_b, z_inputs_b, x2_outputs_b, z_p_outputs_b, z_outputs_b.
                        x_inputs1_b = [x_inputs1['inputs1']]
                        x_inputs2_b = [x_inputs2['inputs2']]
                        
                        if self.nn_arch['label_usage']:
                            z_inputs_b = [np.random.rand(num_samples, self.map_nn_arch['latent_dim'])] \
                                + [np.random.randint(self.map_nn_arch['num_classes'], size=(num_samples, 1))]
                        else:
                            z_inputs_b = [np.random.rand(num_samples, self.map_nn_arch['latent_dim'])]
                            
                        x_outputs_b = [np.ones(shape=tuple([num_samples] + list(self.disc.output_shape[1:])))]
                        z_outputs_b = [np.zeros(shape=tuple([num_samples] + list(self.disc.output_shape[1:])))]
             
                        # Train disc.
                        if self.conf['model_loading']:
                            if self.conf['multi_gpu']:                                     
                                outs1 = self.disc_p.train_on_batch(x_inputs1_b + x_inputs2_b
                                         , x_outputs_b) #?
                                
                                # Create normal random inputs.
                                internal_inputs = []
                                internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                   + list(K.int_shape(self.gen.inputs[2]))[1:])))
                                                            
                                for inp in self.gen.inputs[3:]:
                                    if K.ndim(inp) == 4:
                                        internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                   + list(K.int_shape(inp)[1:]))))
                                    else:
                                        internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                   + list(K.int_shape(inp)[1:])))) # Trivial.
                                
                                z_inputs_b = self.gen_p.predict_on_batch(z_inputs_b + internal_inputs) 
                                outs2 = self.disc_p.train_on_batch(z_inputs_b
                                         , z_outputs_b) #?
                                outs = 0.5 * np.add(outs1, outs2)#?                   
                            else:
                                outs1 = self.disc.train_on_batch(x_inputs1_b + x_inputs2_b
                                         , x_outputs_b) #? 
                                
                                # Create normal random inputs.
                                internal_inputs = []
                                internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                   + list(K.int_shape(self.gen.inputs[2]))[1:])))
                                                            
                                for inp in self.gen.inputs[3:]:
                                    if K.ndim(inp) == 4:
                                        internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                   + list(K.int_shape(inp)[1:]))))
                                    else:
                                        internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                   + list(K.int_shape(inp)[1:])))) # Trivial.
                                
                                z_inputs_b = self.gen.predict_on_batch(z_inputs_b + internal_inputs) 
                                outs2 = self.disc.train_on_batch(z_inputs_b
                                         , z_outputs_b) #?
                                outs = 0.5 * np.add(outs1, outs2)#?                        
                        else:                                                                                    
                            if self.conf['multi_gpu']:
                                outs1 = self.disc_p.train_on_batch(x_inputs1_b + x_inputs2_b
                                         , x_outputs_b) #?
                                                                
                                # Create normal random inputs.
                                internal_inputs = []
                                internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                   + list(K.int_shape(self.gen.inputs[2]))[1:])))
                                                            
                                for inp in self.gen.inputs[3:]:
                                    if K.ndim(inp) == 4:
                                        internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                   + list(K.int_shape(inp)[1:]))))
                                    else:
                                        internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                   + list(K.int_shape(inp)[1:])))) # Trivial.
                                
                                z_inputs_b = self.gen_p.predict_on_batch(z_inputs_b + internal_inputs)  
                                outs2 = self.disc_p.train_on_batch(z_inputs_b
                                         , z_outputs_b) #?
                                outs = 0.5 * np.add(outs1, outs2)#?                    
                            else:
                                outs1 = self.disc.train_on_batch(x_inputs1_b + x_inputs2_b
                                         , x_outputs_b) #? 
                                                                
                                # Create normal random inputs.
                                internal_inputs = []
                                internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                   + list(K.int_shape(self.gen.inputs[2]))[1:])))
                                                            
                                for inp in self.gen.inputs[3:]:
                                    if K.ndim(inp) == 4:
                                        internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                   + list(K.int_shape(inp)[1:]))))
                                    else:
                                        internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                   + list(K.int_shape(inp)[1:])))) # Trivial.
                                
                                z_inputs_b = self.gen.predict_on_batch(z_inputs_b + internal_inputs) 
                                outs2 = self.disc.train_on_batch(z_inputs_b
                                         , z_outputs_b) #?
                                outs = 0.5 * np.add(outs1, outs2)#?  

                        #print(s_i, self.map.get_weights()[0])

                        outs = to_list(outs) + [outs1, outs2] #?
                        
                        metric_names = self.disc.metrics_names + ['real_loss', 'fake_loss'] #?                          
                            
                        for l, o in zip(metric_names, outs):
                            k_batch_logs[l] = o                        
            
                        ws = self.gen.get_weights()
                        res = []
                        for w in ws:
                            res.append(np.isfinite(w).all())
                        res = np.asarray(res)

                        print('\n', k_batch_logs)
                        #callbacks_disc.on_batch_end(self.hps['batch_step'] * s_i + k_i, k_batch_logs)
                        
                    # Build batch logs.
                    batch_logs = {'batch': s_i, 'size': self.hps['mini_batch_size']}
                    callbacks_gan.on_batch_begin(s_i, batch_logs)

                    if self.nn_arch['label_usage']:
                        z_inputs_b = [np.random.rand(num_samples, self.map_nn_arch['latent_dim'])] \
                                + [np.random.randint(self.map_nn_arch['num_classes'], size=(num_samples, 1))]
                    else:
                        z_inputs_b = [np.random.rand(num_samples, self.map_nn_arch['latent_dim'])]
                    
                    z_p_outputs_b = [np.ones(shape=tuple([num_samples] + list(self.disc.output_shape[1:])))]
                    
                    # Train gan.
                    if self.conf['model_loading']:
                        # Create normal random inputs.
                        internal_inputs = []
                        internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                           + list(K.int_shape(self.gan.inputs[2]))[1:])))
                                                    
                        for inp in self.gan.inputs[3:]:
                            if K.ndim(inp) == 4:
                                internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                           + list(K.int_shape(inp)[1:]))))
                            else:
                                internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                           + list(K.int_shape(inp)[1:])))) # Trivial.
                        
                        if self.conf['multi_gpu']:
                            outs = self.gan_p.train_on_batch(z_inputs_b + internal_inputs, z_p_outputs_b)
                        else:
                            outs = self.gan.train_on_batch(z_inputs_b + internal_inputs, z_p_outputs_b)
                    else:
                        # Create normal random inputs.
                        internal_inputs = []
                        internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                           + list(K.int_shape(self.gan.inputs[2]))[1:])))
                                                    
                        for inp in self.gan.inputs[3:]:
                            if K.ndim(inp) == 4:
                                internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                           + list(K.int_shape(inp)[1:]))))
                            else:
                                internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                           + list(K.int_shape(inp)[1:])))) # Trivial.
                        
                        if self.conf['multi_gpu']:
                            outs = self.gan_p.train_on_batch(z_inputs_b + internal_inputs, z_p_outputs_b)
                        else:
                            outs = self.gan.train_on_batch(z_inputs_b + internal_inputs, z_p_outputs_b)

                    outs = to_list(outs)
                    
                    if self.conf['multi_gpu']:
                        for l, o in zip(self.gan_p.metrics_names, outs):
                            batch_logs[l] = o
                    else:
                        for l, o in zip(self.gan.metrics_names, outs):
                            batch_logs[l] = o
        
                    print('\n', batch_logs)
                    #callbacks_gan.on_batch_end(s_i, batch_logs)
                    
                    # Balance training.
                    if k_batch_logs['loss'] > batch_logs['loss'] and np.abs(k_batch_logs['loss'] - batch_logs['loss']) > 0.1:
                        while np.abs(k_batch_logs['loss'] - batch_logs['loss']) > 0.1:
                            for k_i in range(self.hps['disc_k_step']): #?
                                # Build batch logs.
                                k_batch_logs = {'batch': self.hps['batch_step'] * s_i + k_i, 'size': self.hps['mini_batch_size']}
                                #callbacks_disc.on_batch_begin(self.hps['batch_step'] * s_i + k_i, k_batch_logs)
                                
                                x_inputs1, x_inputs2 = next(output_generator)
                                
                                # Create x_inputs_b, z_inputs_b, x2_outputs_b, z_p_outputs_b, z_outputs_b.
                                x_inputs1_b = [x_inputs1['inputs1']]
                                x_inputs2_b = [x_inputs2['inputs2']]
                                
                                if self.nn_arch['label_usage']:
                                    z_inputs_b = [np.random.rand(num_samples, self.map_nn_arch['latent_dim'])] \
                                        + [np.random.randint(self.map_nn_arch['num_classes'], size=(num_samples, 1))]
                                else:
                                    z_inputs_b = [np.random.rand(num_samples, self.map_nn_arch['latent_dim'])]
                                    
                                x_outputs_b = [np.ones(shape=tuple([num_samples] + list(self.disc.output_shape[1:])))]
                                z_outputs_b = [np.zeros(shape=tuple([num_samples] + list(self.disc.output_shape[1:])))]
                     
                                # Train disc.
                                if self.conf['model_loading']:
                                    if self.conf['multi_gpu']:                                     
                                        outs1 = self.disc_p.train_on_batch(x_inputs1_b + x_inputs2_b
                                                 , x_outputs_b) #?
                                        
                                        # Create normal random inputs.
                                        internal_inputs = []
                                        internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                           + list(K.int_shape(self.gen.inputs[2]))[1:])))
                                                                    
                                        for inp in self.gen.inputs[3:]:
                                            if K.ndim(inp) == 4:
                                                internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                           + list(K.int_shape(inp)[1:]))))
                                            else:
                                                internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                           + list(K.int_shape(inp)[1:])))) # Trivial.
                                        
                                        z_inputs_b = self.gen_p.predict_on_batch(z_inputs_b + internal_inputs) 
                                        outs2 = self.disc_p.train_on_batch(z_inputs_b
                                                 , z_outputs_b) #?
                                        outs = 0.5 * np.add(outs1, outs2)#?                   
                                    else:
                                        outs1 = self.disc.train_on_batch(x_inputs1_b + x_inputs2_b
                                                 , x_outputs_b) #? 
                                        
                                        # Create normal random inputs.
                                        internal_inputs = []
                                        internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                           + list(K.int_shape(self.gen.inputs[2]))[1:])))
                                                                    
                                        for inp in self.gen.inputs[3:]:
                                            if K.ndim(inp) == 4:
                                                internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                           + list(K.int_shape(inp)[1:]))))
                                            else:
                                                internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                           + list(K.int_shape(inp)[1:])))) # Trivial.
                                        
                                        z_inputs_b = self.gen.predict_on_batch(z_inputs_b + internal_inputs) 
                                        outs2 = self.disc.train_on_batch(z_inputs_b
                                                 , z_outputs_b) #?
                                        outs = 0.5 * np.add(outs1, outs2)#?                        
                                else:                                                                                    
                                    if self.conf['multi_gpu']:
                                        outs1 = self.disc_p.train_on_batch(x_inputs1_b + x_inputs2_b
                                                 , x_outputs_b) #?
                                                                        
                                        # Create normal random inputs.
                                        internal_inputs = []
                                        internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                           + list(K.int_shape(self.gen.inputs[2]))[1:])))
                                                                    
                                        for inp in self.gen.inputs[3:]:
                                            if K.ndim(inp) == 4:
                                                internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                           + list(K.int_shape(inp)[1:]))))
                                            else:
                                                internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                           + list(K.int_shape(inp)[1:])))) # Trivial.
                                        
                                        z_inputs_b = self.gen_p.predict_on_batch(z_inputs_b + internal_inputs)  
                                        outs2 = self.disc_p.train_on_batch(z_inputs_b
                                                 , z_outputs_b) #?
                                        outs = 0.5 * np.add(outs1, outs2)#?                    
                                    else:
                                        outs1 = self.disc.train_on_batch(x_inputs1_b + x_inputs2_b
                                                 , x_outputs_b) #? 
                                                                        
                                        # Create normal random inputs.
                                        internal_inputs = []
                                        internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                           + list(K.int_shape(self.gen.inputs[2]))[1:])))
                                                                    
                                        for inp in self.gen.inputs[3:]:
                                            if K.ndim(inp) == 4:
                                                internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                           + list(K.int_shape(inp)[1:]))))
                                            else:
                                                internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                           + list(K.int_shape(inp)[1:])))) # Trivial.
                                        
                                        z_inputs_b = self.gen.predict_on_batch(z_inputs_b + internal_inputs) 
                                        outs2 = self.disc.train_on_batch(z_inputs_b
                                                 , z_outputs_b) #?
                                        outs = 0.5 * np.add(outs1, outs2)#?  
        
                                for d in internal_inputs:
                                    del d
                                #print(s_i, self.map.get_weights()[0])
        
                                outs = to_list(outs) + [outs1, outs2] #?
                                
                                metric_names = self.disc.metrics_names + ['real_loss', 'fake_loss'] #?                          
                                    
                                for l, o in zip(metric_names, outs):
                                    k_batch_logs[l] = o                        
                    
                                ws = self.gen.get_weights()
                                res = []
                                for w in ws:
                                    res.append(np.isfinite(w).all())
                                res = np.asarray(res)
        
                                print('\n', k_batch_logs)
                                #callbacks_disc.on_batch_end(self.hps['batch_step'] * s_i + k_i, k_batch_logs)
                    elif k_batch_logs['loss'] < batch_logs['loss'] and np.abs(k_batch_logs['loss'] - batch_logs['loss']) > 0.1:
                        while np.abs(k_batch_logs['loss'] - batch_logs['loss']) > 0.1:
                            # Build batch logs.
                            batch_logs = {'batch': s_i, 'size': self.hps['mini_batch_size']}
                            #callbacks_gan.on_batch_begin(s_i, batch_logs)
        
                            if self.nn_arch['label_usage']:
                                z_inputs_b = [np.random.rand(num_samples, self.map_nn_arch['latent_dim'])] \
                                        + [np.random.randint(self.map_nn_arch['num_classes'], size=(num_samples, 1))]
                            else:
                                z_inputs_b = [np.random.rand(num_samples, self.map_nn_arch['latent_dim'])]
                            
                            z_p_outputs_b = [np.ones(shape=tuple([num_samples] + list(self.disc.output_shape[1:])))]
                            
                            # Train gan.
                            if self.conf['model_loading']:
                                # Create normal random inputs.
                                internal_inputs = []
                                internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                   + list(K.int_shape(self.gan.inputs[2]))[1:])))
                                                            
                                for inp in self.gan.inputs[3:]:
                                    if K.ndim(inp) == 4:
                                        internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                   + list(K.int_shape(inp)[1:]))))
                                    else:
                                        internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                   + list(K.int_shape(inp)[1:])))) # Trivial.
                                
                                if self.conf['multi_gpu']:
                                    outs = self.gan_p.train_on_batch(z_inputs_b + internal_inputs, z_p_outputs_b)
                                else:
                                    outs = self.gan.train_on_batch(z_inputs_b + internal_inputs, z_p_outputs_b)
                            else:
                                # Create normal random inputs.
                                internal_inputs = []
                                internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                   + list(K.int_shape(self.gan.inputs[2]))[1:])))
                                                            
                                for inp in self.gan.inputs[3:]:
                                    if K.ndim(inp) == 4:
                                        internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                   + list(K.int_shape(inp)[1:]))))
                                    else:
                                        internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                                                   + list(K.int_shape(inp)[1:])))) # Trivial.
                                
                                if self.conf['multi_gpu']:
                                    outs = self.gan_p.train_on_batch(z_inputs_b + internal_inputs, z_p_outputs_b)
                                else:
                                    outs = self.gan.train_on_batch(z_inputs_b + internal_inputs, z_p_outputs_b)
        
                            for d in internal_inputs:
                                del d
                                    
                            outs = to_list(outs)
                            
                            if self.conf['multi_gpu']:
                                for l, o in zip(self.gan_p.metrics_names, outs):
                                    batch_logs[l] = o
                            else:
                                for l, o in zip(self.gan.metrics_names, outs):
                                    batch_logs[l] = o
                
                            print('\n', batch_logs)
                            #callbacks_gan.on_batch_end(s_i, batch_logs)                            
                              
                callbacks_disc.on_epoch_end(e_i, epochs_log)
                callbacks_gan.on_epoch_end(e_i, epochs_log)
        
            callbacks_disc.on_train_end() #?
            callbacks_gan.on_train_end() #?  
        finally:
            try:
                if enq is not None:
                    enq.stop()
            finally:
                pass

        # Save models.
        with CustomObjectScope(self.custom_objects):
            self.disc.save(self.DISC_PATH)
            self.gan.save(self.GAN_PATH)

        if self.conf['multi_gpu']:
            return self.disc_p.history, self.gan_p.history
        else:
            return self.disc.history, self.gan.history

    def generate(self, latents, labels, *args, **kwargs):
        """Generate styled images.
        
        Parameters
        ----------
        latents: 2d numpy array
            latents.
        labels: 2d numpy array
            Labels.
        """ 
        super(StyleGAN, self).generate(self, *args, **kwargs) #?
        
        num_samples = latents.shape[0]
        if self.conf['model_loading']: #?
            # Create normal random inputs.
            internal_inputs = []
            internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                               + list(K.int_shape(self.disc_ext.inputs[4]))[1:])))
                                        
            for inp in self.disc_ext.inputs[5:]:
                if K.ndim(inp) == 4:
                    internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                               + list(K.int_shape(inp)[1:]))))
                else:
                    internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                               + list(K.int_shape(inp)[1:])))) # Trivial.
   
            if self.conf['multi_gpu']:
                if self.nn_arch['label_usage']:
                    s_images = self.gen_p.predict([latents, labels] + internal_inputs)
                else:
                    s_images = self.gen_p.predict([latents] + internal_inputs)
            else:
                if self.nn_arch['label_usage']:
                    s_images = self.gen.predict([latents, labels] + internal_inputs)
                else:
                    s_images = self.gen.predict([latents] + internal_inputs)
        else:                
            # Create normal random inputs.
            internal_inputs = []
            internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                               + list(K.int_shape(self.disc_ext.inputs[4]))[1:])))
                                        
            for inp in self.disc_ext.inputs[5:]:
                if K.ndim(inp) == 4:
                    internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                               + list(K.int_shape(inp)[1:]))))
                else:
                    internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                               + list(K.int_shape(inp)[1:])))) # Trivial.
   
            if self.conf['multi_gpu']:
                if self.nn_arch['label_usage']:
                    s_images = self.gen_p.predict([latents, labels] + internal_inputs)
                else:
                    s_images = self.gen_p.predict([latents] + internal_inputs)
            else:
                if self.nn_arch['label_usage']:
                    s_images = self.gen.predict([latents, labels] + internal_inputs)
                else:
                    s_images = self.gen.predict([latents] + internal_inputs)
        
        return s_images

    def generate2(self, latents, labels, *args, **kwargs):
        """Generate styled images.
        
        Parameters
        ----------
        latents: 2d numpy array
            latents.
        labels: 2d numpy array
            Labels.
        """ 
        super(StyleGAN, self).generate(self, *args, **kwargs) #?
        
        num_samples = latents.shape[0]
        if self.conf['model_loading']: #?
            # Create normal random inputs.
            internal_inputs = []
            internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                               + list(K.int_shape(self.gen.inputs[2]))[1:])))
                                        
            for inp in self.gen.inputs[3:]:
                if K.ndim(inp) == 4:
                    internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                               + list(K.int_shape(inp)[1:]))))
                else:
                    internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                               + list(K.int_shape(inp)[1:])))) # Trivial.
   
            if self.conf['multi_gpu']:
                if self.nn_arch['label_usage']:
                    s_images = self.gen_p.predict([latents, labels] + internal_inputs)
                else:
                    s_images = self.gen_p.predict([latents] + internal_inputs)
            else:
                if self.nn_arch['label_usage']:
                    s_images = self.gen.predict([latents, labels] + internal_inputs)
                else:
                    s_images = self.gen.predict([latents] + internal_inputs)
        else:                
            # Create normal random inputs.
            internal_inputs = []
            internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                               + list(K.int_shape(self.gen.inputs[2]))[1:])))
                                        
            for inp in self.gen.inputs[3:]:
                if K.ndim(inp) == 4:
                    internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                               + list(K.int_shape(inp)[1:]))))
                else:
                    internal_inputs.append(np.random.normal(size=tuple([num_samples] \
                                                               + list(K.int_shape(inp)[1:])))) # Trivial.
   
            if self.conf['multi_gpu']:
                if self.nn_arch['label_usage']:
                    s_images = self.gen_p.predict([latents, labels] + internal_inputs)
                else:
                    s_images = self.gen_p.predict([latents] + internal_inputs)
            else:
                if self.nn_arch['label_usage']:
                    s_images = self.gen.predict([latents, labels] + internal_inputs)
                else:
                    s_images = self.gen.predict([latents] + internal_inputs)
        
        return s_images
                
def main():
    """Main."""
    
    # Load configuration.
    if platform.system() == 'Windows':
        with open(os.path.join("style_based_gan_conf_win.json"), 'r') as f:
            conf = json.load(f)  
    else:
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