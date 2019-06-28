"""
Created on 2019. 5. 13.

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
from keras.layers import Input, Dense, Lambda
from keras.utils import multi_gpu_model
from keras import optimizers

from ku.ebm.rbm import RBM 

#os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# Constants.
DEBUG = True
MULTI_GPU = False
NUM_GPUS = 4

class MNISTClassifier(object):
    """MNIST digit classifier using the RBM + Softmax model."""
    # Constants.
    MODEL_PATH = 'digit_classificaton_model.h5'
    IMAGE_SIZE = 784
    
    def __init__(self, conf):
        self.conf = conf
            
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']
        self.model_loading = self.conf['model_loading']

        if self.model_loading: 
            if MULTI_GPU:
                self.digit_classificaton_model = load_model(self.MODEL_PATH, custom_objects={'RBM': RBM})
                self.rbm = self.digit_classificaton_model.get_layer('rbm_1')
                
                self.digit_classificaton_parallel_model = multi_gpu_model(self.model, gpus = NUM_GPUS)
                opt = optimizers.Adam(lr=self.hps['lr']
                                        , beta_1=self.hps['beta_1']
                                        , beta_2=self.hps['beta_2']
                                        , decay=self.hps['decay']) 
                self.digit_classificaton_parallel_model.compile(optimizer=opt, loss='mse') 
            else:
                self.digit_classificaton_model = load_model(self.MODEL_PATH, custom_objects={'RBM': RBM})
                self.digit_classificaton_model.summary()
                self.rbm = self.digit_classificaton_model.get_layer('rbm_1')
        else:        
            # Design the model.
            input_image = Input(shape=(self.IMAGE_SIZE,))
            x = Lambda(lambda x: x/255)(input_image)
            
            # RBM layer.
            self.rbm = RBM(self.conf['rbm_hps'], self.nn_arch['output_dim'], name='rbm') # Name?
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
        print('Train the RBM model.')
        self.rbm.fit(V)
        
        # Supervised learning.
        print('Train the NN model.')
        if MULTI_GPU:
            self.digit_classificaton_parallel_model.fit(V
                                           , gt
                                           , batch_size=self.hps['batch_size']
                                           , epochs=self.hps['epochs']
                                           , verbose=1)        
        else:
            self.digit_classificaton_model.fit(V
                                           , gt
                                           , batch_size=self.hps['batch_size']
                                           , epochs=self.hps['epochs']
                                           , verbose=1)

        print('Save the model.')            
        self.digit_classificaton_model.save(self.MODEL_PATH)
    
    def _load_training_data(self):
        """Load training data."""
        train_df = pd.read_csv('train.csv')
        V = []
        gt = []
        
        for i in range(train_df.shape[0]):
            V.append(train_df.iloc[i, 1:].values/255)
            t_gt = np.zeros(shape=(10,))
            t_gt[train_df.iloc[i,0]] = 1.
            gt.append(t_gt)
        
        V = np.asarray(V, dtype=np.float32)
        gt = np.asarray(gt, dtype=np.float32)
        
        return V, gt
    
    def test(self):
        """Test."""
        # Load test data.
        V = self._load_test_data()
        
        # Predict digits.
        res = self.digit_classificaton_model.predict(V
                                                     , verbose=1)
        
        # Record results into a file.
        with open('solution.csv', 'w') as f:
            f.write('ImageId,Label\n')
            
            for i, v in enumerate(res):
                f.write(str(i + 1) + ',' + str(np.argmax(v)) + '\n') 
        
    def _load_test_data(self):
        """Load test data."""
        test_df = pd.read_csv('test.csv')
        V = []
        
        for i in range(test_df.shape[0]):
            V.append(test_df.iloc[i, :].values/255)
        
        V = np.asarray(V, dtype=np.float32)
        
        return V       

def main():
    """Main."""
    
    # Load configuration.
    with open(os.path.join("rbm_softmax_mnist_conf.json"), 'r') as f:
        conf = json.load(f)

    if conf['mode'] == 'train':      
        # Train.
        mc = MNISTClassifier(conf)
        
        ts = time.time()
        mc.train()
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
    elif conf['mode'] == 'test':      
        # Test.
        mc = MNISTClassifier(conf)      
        
        ts = time.time()
        mc.test()
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
        
if __name__ == '__main__':    
    main()