"""
Created on 2019. 5. 13.

@author: Inwoo Chung (gutomitai@gmail.com)
"""

import os
import glob
import argparse
import time
import pickle
import platform
import shutil

import numpy as np
import pandas as pd
import cv2 as cv
from skimage.io import imsave
from scipy.stats import entropy
from scipy.linalg import norm

from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, ZeroPadding2D, LeakyReLU, Flatten, Concatenate
from keras.layers.merge import add
from keras.utils import multi_gpu_model
from keras.utils.data_utils import Sequence
import keras.backend as K
from keras import optimizers

from ku.ebm.rbm import RBM 

class MNISTClassifier(object):
    """MNIST digit classifier using the RBM + Softmax model."""
    
    # Constants.
    IMAGE_SIZE = 28 # Symmetry.
    
    def __init__(self, hps, nn_arch_info, model_loading=False):
        self.hps = hps
        self.nn_arch_info = nn_arch_info
        
        # Design the model.
        input_image = Input(shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 1), dtype=np.float32)
        x = input_image/255
        
        # RBM layer.
        self.rbm = RBM(self.hps['rbm_hps'], self.nn_arch_info['output_dim'])
        x = Flatten()(x)
        x = self.rbm(x) #?
        
        # Softmax layer.
        output = Dense(10, activation='softmax')(x)
        
        # Create a model.
        self.digit_classificaton_model = Model(inputs=[input_image], outputs=[output])
        
        opt = optimizers.Adam(lr=self.hps['lr']
                                , beta_1=self.hps['beta_1']
                                , beta_2=self.hps['beta_2']
                                , decay=self.hps['decay'])
        
        self.digit_classificaton_model.compile(optimizer=opt, loss='categorical_crossentropy')
        self.digit_classificaton_model.summary() 

    def train(self):
        """Train."""
        # Load training data.
        V, gt = self._load_training_data()
        
        # Semi-supervised learning.
        # Unsupervised learning.
        # RBM training.
        self.rbm.fit(V)
        
        # Supervised learning.
        self.digit_classificaton_model.fit(V
                                           , gt
                                           , batch_size=self.hps['batch_size']
                                           , epochs=self.hps['epochs']
                                           , verbose=1)
    
    def _load_training_data(self):
        """Load training data."""
        train_df = pd.read_csv('train.csv')
        V = []
        gt = []
        
        for i in range(train_df.shape[0]):
            V.append(train_df.iloc[i, 1:].values)
            gt.append(np.asarray([train_df.iloc[i,0]]))
        
        V = np.asarray(V, dtype=np.float32)
        gt = np.asarray(gt, dtype=np.int32)
        
        return V, gt

def main(args):
    """Main.
    
    Parameters
    ----------
    args : argument type 
        Arguments
    """
    hps = {}
    nn_arch_info = {}

    if args.mode == 'train':
        # Get arguments.      
        nn_arch_info['output_dim'] = int(args.output_dim)   
        
        hps['lr'] = float(args.lr)
        hps['beta_1'] = float(args.beta_1)
        hps['beta_2'] = float(args.beta_2)
        hps['decay'] = float(args.decay)
        hps['batch_size'] = int(args.batch_size)
        hps['epochs'] = int(args.epochs)
        
        rbm_hps = {}
        rbm_hps['lr'] = float(args.rbm_lr)
        rbm_hps['batch_size'] = int(args.rbm_batch_size)
        rbm_hps['epochs'] = int(args.rbm_epochs)
        hps['rbm_hps'] = rbm_hps        
        
        model_loading = False if int(args.model_loading) == 0 else True        
        
        # Train.
        mc = MNISTClassifier(hps, nn_arch_info, model_loading)
        
        ts = time.time()
        mc.train()
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
        
if __name__ == '__main__':
    # Parse arguments.
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode')
    parser.add_argument('--output_dim')
    parser.add_argument('--lr')
    parser.add_argument('--beta_1')
    parser.add_argument('--beta_2')
    parser.add_argument('--decay')
    parser.add_argument('--batch_size')
    parser.add_argument('--epochs')
    parser.add_argument('--rbm_lr')
    parser.add_argument('--rbm_batch_size')
    parser.add_argument('--rbm_epochs')    
    parser.add_argument('--model_loading')
    args = parser.parse_args()
    
    main(args)