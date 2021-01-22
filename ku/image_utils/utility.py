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
from scipy import ndimage
from cupyx.scipy import ndimage as ndimage_gpu
import cupy as cp

# Constants.
DEVICE_CPU = -1


def resize(image, size: tuple, mode='constant', device: int=DEVICE_CPU):
    '''Resize image using scipy's affine_transform.

    Parameters
    ----------
    image: 3d numpy array or cypy array.
        Image data.
    size: Tuple.
        Target width and height.
    mode: String.
        Boundary mode (default is constant).
    device: Integer.
        Device kind (default is cpu).

    Returns
    -------
    3d numpy or cypy array.
        Resized image data.
    '''

    # Calculate x, y scaling factors and output shape.
    w, h = size
    h_o, w_o, _ = image.shape
    fx = w / float(w_o)
    fy = h / float(h_o)
    output_shape = (h, w, image.shape[2])

    # Calculate resizing according to device.
    if device == DEVICE_CPU:
        # Create affine transformation matrix.
        M = np.eye(4)
        M[0,0] = 1.0 / fy
        M[1,1] = 1.0 / fx
        M = M[0:3]

        # Resize.
        resized_image = ndimage.affine_transform(image
                                                 , M
                                                 , output_shape=output_shape
                                                 , mode=mode)

        return resized_image
    elif device > DEVICE_CPU:
        if hasattr(image, 'device') != True:
            image_gpu = cp.asarray(image)
        else:
            image_gpu = image

        # Create affine transformation matrix.
        M_gpu = cp.eye(4)
        M_gpu[0, 0] = 1.0 / fy
        M_gpu[1, 1] = 1.0 / fx
        M_gpu = M_gpu[0:3]

        # Resize.
        resized_image = ndimage_gpu.affine_transform(image_gpu
                                                     , M_gpu
                                                     , output_shape=output_shape
                                                     , mode=mode)

        if hasattr(image, 'device') != True:
            return resized_image.get()
        else:
            return resized_image
    else:
        raise ValueError('device is not valid.')


def resize_image_to_target_symmeric_size(image, size: int, device=DEVICE_CPU):
    """Resize image to target symmetric size.
    
    Parameters
    ----------
    image: 3d numpy array or cypy array.
        Image data.
    size: Integer.
        Target symmetric image size.
    device: Integer.
        Device kind (default is cpu).

    Returns
    -------
    3d numpy or cypy array.
        Resized image data.
    """

    # Adjust the original image size into the normalized image size according to the ratio of width, height.
    w = image.shape[1]
    h = image.shape[0]
    pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0
                    
    if w >= h:
        w_p = size
        h_p = int(h / w * size)
        pad = size - h_p
        
        if pad % 2 == 0:
            pad_t = pad // 2
            pad_b = pad // 2
        else:
            pad_t = pad // 2
            pad_b = pad // 2 + 1

        image_p = resize(image, (w_p, h_p), mode='nearest', device=device)

        if device == DEVICE_CPU:
            image_p = np.pad(image_p, ((pad_t, pad_b), (0, 0), (0, 0)))
        elif device > DEVICE_CPU:
            if hasattr(image, 'device') != True:
                image_gpu = cp.asarray(image_p)
            else:
                image_gpu = image_p

            image_gpu = cp.pad(image_gpu, ((pad_t, pad_b), (0, 0), (0, 0)))

            if hasattr(image, 'device') != True: #?
                image_p = image_gpu.get()
            else:
                image_p = image_gpu
    else:
        h_p = size
        w_p = int(w / h * size)
        pad = size - w_p
        
        if pad % 2 == 0:
            pad_l = pad // 2
            pad_r = pad // 2
        else:
            pad_l = pad // 2
            pad_r = pad // 2 + 1                
        
        image_p = resize(image, (w_p, h_p), mode='nearest', device=device)

        if device == DEVICE_CPU:
            image_p = np.pad(image_p, ((0, 0), (pad_r, pad_l), (0, 0)))
        elif device > DEVICE_CPU:
            if hasattr(image, 'device') != True:
                image_gpu = cp.asarray(image_p)
            else:
                image_gpu = image_p

            image_gpu = cp.pad(image_gpu, ((0, 0), (pad_r, pad_l), (0, 0)))

            if hasattr(image, 'device') != True: #?
                image_p = image_gpu.get()
            else:
                image_p = image_gpu

    return image_p, w, h, pad_t, pad_l, pad_b, pad_r

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
