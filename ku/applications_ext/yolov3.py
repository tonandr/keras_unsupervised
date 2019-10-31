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
import cv2 as cv
from skimage.io import imread, imsave
from scipy.linalg import norm
import h5py

from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Input, Dense, Lambda, ZeroPadding2D
from tensorflow.python.keras.layers import LeakyReLU, Flatten, Concatenate, Reshape, ReLU
from tensorflow.python.keras.layers import Conv2DTranspose, BatchNormalization 
from tensorflow.python.keras.layers.merge import add, subtract
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.utils.data_utils import Sequence
import tensorflow.python.keras.backend as K
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine.input_layer import InputLayer

from .yolov3_detect import make_yolov3_model, BoundBox, WeightReader, draw_boxes_v3

def YOLOV3(self):
    """Get yolov3.
    
    Returns
    -------
    Model of Keras
        Partial yolo3 model from the input layer to the add_23 layer
    """
        
    yolov3 = make_yolov3_model()

    # Load the weights.
    weight_reader = WeightReader('yolov3.weights')
    weight_reader.load_weights(yolov3)
    
    # Make a base model.
    input1 = Input(shape=(self.nn_arch['image_size'], self.nn_arch['image_size'], 3), name='input1')
    
    # 0 ~ 1.
    conv_layer = yolov3.get_layer('conv_' + str(0))
    x = ZeroPadding2D(1)(input1) #?               
    x = conv_layer(x)
    norm_layer = yolov3.get_layer('bnorm_' + str(0))
    x = norm_layer(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    conv_layer = yolov3.get_layer('conv_' + str(1))
    x = ZeroPadding2D(1)(x) #?               
    x = conv_layer(x)
    norm_layer = yolov3.get_layer('bnorm_' + str(1))
    x = norm_layer(x)
    x = LeakyReLU(alpha=0.1)(x)
    skip = x
    
    # 2 ~ 3.
    for i in range(2, 4, 2):
        conv_layer = yolov3.get_layer('conv_' + str(i))
        
        if conv_layer.kernel_size[0] > 1:
            x = ZeroPadding2D(1)(x) #? 
          
        x = conv_layer(x)
        norm_layer = yolov3.get_layer('bnorm_' + str(i))
        x = norm_layer(x)
        x = LeakyReLU(alpha=0.1)(x)

        conv_layer = yolov3.get_layer('conv_' + str(i + 1))
        
        if conv_layer.kernel_size[0] > 1:
            x = ZeroPadding2D(1)(x) #? 
          
        x = conv_layer(x)
        norm_layer = yolov3.get_layer('bnorm_' + str(i + 1))
        x = norm_layer(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = add([skip, x]) #?

    # 5.
    conv_layer = yolov3.get_layer('conv_' + str(5))
    
    if conv_layer.kernel_size[0] > 1:
        x = ZeroPadding2D(1)(x) #? 
      
    x = conv_layer(x)
    norm_layer = yolov3.get_layer('bnorm_' + str(5))
    x = norm_layer(x)
    x = LeakyReLU(alpha=0.1)(x)
    skip = x
    
    # 6 ~ 10.
    for i in range(6, 10, 3):
        conv_layer = yolov3.get_layer('conv_' + str(i))
        
        if conv_layer.kernel_size[0] > 1:
            x = ZeroPadding2D(1)(x) #? 
          
        x = conv_layer(x)
        norm_layer = yolov3.get_layer('bnorm_' + str(i))
        x = norm_layer(x)
        x = LeakyReLU(alpha=0.1)(x)

        conv_layer = yolov3.get_layer('conv_' + str(i + 1))
        
        if conv_layer.kernel_size[0] > 1:
            x = ZeroPadding2D(1)(x) #? 
          
        x = conv_layer(x)
        norm_layer = yolov3.get_layer('bnorm_' + str(i + 1))
        x = norm_layer(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = add([skip, x]) #?
        skip = x #?

    # 12.
    conv_layer = yolov3.get_layer('conv_' + str(12))
    
    if conv_layer.kernel_size[0] > 1:
        x = ZeroPadding2D(1)(x) #? 
      
    x = conv_layer(x)
    norm_layer = yolov3.get_layer('bnorm_' + str(12))
    x = norm_layer(x)
    x = LeakyReLU(alpha=0.1)(x)
    skip = x

    # 13 ~ 35.
    for i in range(13, 35, 3):
        conv_layer = yolov3.get_layer('conv_' + str(i))
        
        if conv_layer.kernel_size[0] > 1:
            x = ZeroPadding2D(1)(x) #? 
          
        x = conv_layer(x)
        norm_layer = yolov3.get_layer('bnorm_' + str(i))
        x = norm_layer(x)
        x = LeakyReLU(alpha=0.1)(x)

        conv_layer = yolov3.get_layer('conv_' + str(i + 1))
        
        if conv_layer.kernel_size[0] > 1:
            x = ZeroPadding2D(1)(x) #? 
          
        x = conv_layer(x)
        norm_layer = yolov3.get_layer('bnorm_' + str(i + 1))
        x = norm_layer(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = add([skip, x]) #?
        skip = x #?

    # 37.
    conv_layer = yolov3.get_layer('conv_' + str(37))
    
    if conv_layer.kernel_size[0] > 1:
        x = ZeroPadding2D(1)(x) #? 
      
    x = conv_layer(x)
    norm_layer = yolov3.get_layer('bnorm_' + str(37))
    x = norm_layer(x)
    x = LeakyReLU(alpha=0.1)(x)
    skip = x

    # 38 ~ 60.
    for i in range(38, 60, 3):
        conv_layer = yolov3.get_layer('conv_' + str(i))
        
        if conv_layer.kernel_size[0] > 1:
            x = ZeroPadding2D(1)(x) #? 
          
        x = conv_layer(x)
        norm_layer = yolov3.get_layer('bnorm_' + str(i))
        x = norm_layer(x)
        x = LeakyReLU(alpha=0.1)(x)

        conv_layer = yolov3.get_layer('conv_' + str(i + 1))
        
        if conv_layer.kernel_size[0] > 1:
            x = ZeroPadding2D(1)(x) #? 
          
        x = conv_layer(x)
        norm_layer = yolov3.get_layer('bnorm_' + str(i + 1))
        x = norm_layer(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = add([skip, x]) #?
        skip = x #?

    # 62.
    conv_layer = yolov3.get_layer('conv_' + str(62))
    
    if conv_layer.kernel_size[0] > 1:
        x = ZeroPadding2D(1)(x) #? 
      
    x = conv_layer(x)
    norm_layer = yolov3.get_layer('bnorm_' + str(62))
    x = norm_layer(x)
    x = LeakyReLU(alpha=0.1)(x)
    skip = x

    # 63 ~ 73.
    for i in range(63, 73, 3):
        conv_layer = yolov3.get_layer('conv_' + str(i))
        
        if conv_layer.kernel_size[0] > 1:
            x = ZeroPadding2D(1)(x) #? 
          
        x = conv_layer(x)
        norm_layer = yolov3.get_layer('bnorm_' + str(i))
        x = norm_layer(x)
        x = LeakyReLU(alpha=0.1)(x)

        conv_layer = yolov3.get_layer('conv_' + str(i + 1))
        
        if conv_layer.kernel_size[0] > 1:
            x = ZeroPadding2D(1)(x) #? 
          
        x = conv_layer(x)
        norm_layer = yolov3.get_layer('bnorm_' + str(i + 1))
        x = norm_layer(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = add([skip, x]) #?
        skip = x #?
    
    output = x
    base = Model(inputs=[input1], outputs=[output])
    base.trainable = True
    base.save('yolov3_base.h5')
    
    return base
