from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import platform
import json
import warnings
import glob
import shutil
import functools
import math

import numpy as np
import pandas as pd
from skimage.io import imread, imsave
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image

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
        image = cv.copyMakeBorder(image, pad_t, pad_b, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0]) 
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
        image = cv.copyMakeBorder(image, 0, 0, pad_l, pad_r, cv.BORDER_CONSTANT, value=[0, 0, 0])
    
    return image

def get_one_hot(inputs, num_classes):
    """Get one hot tensor.
    
    Parameters
    ----------
    inputs: 3d numpy array (a x b x 1) 
        Input array.
    num_classes: integer
        Number of classes.
    
    Returns
    -------
    One hot tensor.
        3d numpy array (a x b x n).
    """
    onehots = np.zeros(shape=tuple(list(inputs.shape[:-1]) + [num_classes]))
    
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            try:
                onehots[i, j, inputs[i, j, 0]] = 1.0
            except IndexError:
                onehots[i, j, 0] = 1.0
        
    return onehots
