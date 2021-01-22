'''
Created on Sep 13, 2020
@author: Inwoo Chung (gutomitai@gmail.com)
'''

import os
import glob
import time
import platform
import shutil
import json
import random

import numpy as np
import pandas as pd
import cv2 as cv
from skimage.io import imread, imsave
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.linalg import norm
import h5py
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from skimage.transform import rotate, rescale
from skimage import exposure

import tensorflow
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Conv3D, ZeroPadding2D, LeakyReLU, Lambda, Concatenate
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, GlobalAveragePooling3D, UpSampling3D
from tensorflow.keras.layers import Activation, Reshape, Add, Multiply, Dropout, UpSampling2D, Dense, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from tensorflow.keras.applications import Xception
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.keras.metrics import AUC

from ku.layer_ext import SeparableConv3D

# Constants.
DEBUG = True


class NobodyConvNet3D(Model):
    """3D convolution network model."""

    def __init__(self, conf, input_shape):
        """
        Parameters
        ----------
        conf: Dictionary
            Configuration dictionary.
        """

        # Initialize.
        self.conf = conf
        self.raw_data_path = self.conf['raw_data_path']
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']
        self.model_loading = self.conf['model_loading']
        self.cell_sizes = [int(self.nn_arch['cell_size'] * 2 ** i) for i in range(self.nn_arch['anchor_scale_size'])]
        self.cell_image_sizes = [int(self.nn_arch['image_size'] / self.cell_sizes[i]) \
                                 for i in range(self.nn_arch['anchor_scale_size'])]

        super(NobodyConvNet3D, self).__init__()

        # Design layers.
        # Start stem.
        nc = int(input_shape[-1] * 10)
        rate = (1, 1, 1)

        self.sep_conv3d_1 = SeparableConv3D(nc
                            , kernel_size=3
                            , depth_multiplier=1
                            , dilation_rate=(rate[0] * self.nn_arch['conv_rate_multiplier']
                                             , rate[1] * self.nn_arch['conv_rate_multiplier']
                                             , rate[2] * self.nn_arch['conv_rate_multiplier'])
                            , padding='same'
                            , use_bias=False
                            , kernel_initializer=initializers.TruncatedNormal())
        self.bn_1 = BatchNormalization(momentum=self.hps['bn_momentum'], scale=self.hps['bn_scale'])
        self.act_1 = Activation('relu')

        # Sequence1.
        rate = (1, 1, 1)
        self.block1_seq1 = Block1(conf, rate, nc)
        nc = int(nc * 1.5)

        # Sequence2.
        rate = (1, 1, 1)
        self.block1_seq2 = Block1(conf, rate, nc)
        nc = int(nc * 1.5)

        # Sequence3.
        rate = (1, 1, 1)
        self.block1_seq3 = Block1(conf, rate, nc)
        nc = int(nc * 1.5)

        # Sequence4.
        rate = (1, 1, 1)
        self.block1_seq4 = Block1(conf, rate, nc)
        nc = int(nc * 1.5)

        rate = (1, 1, 1)
        self.block2_seq4 = Block2(conf, rate, nc)

        # Sequence5.
        rate = (1, 1, 1)
        self.block1_seq5 = Block1(conf, rate, nc)
        nc = int(nc * 1.5)

        rate = (1, 1, 1)
        self.block2_seq5 = Block2(conf, rate, nc)
        self.block2_seq5_2 = Block2(conf, rate, nc)

        # Sequence6.
        rate = (1, 1, 1)
        self.block1_seq6 = Block1(conf, rate, nc)
        nc = int(nc * 1.5)

        rate = (1, 1, 1)
        self.block2_seq6 = Block2(conf, rate, nc)
        self.block2_seq6_2 = Block2(conf, rate, nc)

        # Final stem 1.
        self.module5 = Module5(conf, self.nn_arch['sp_feature_dim'])

    def call(self, input_tensor):
        x = self.sep_conv3d_1(input_tensor)
        x = self.bn_1(x)
        x = self.act_1(x)

        x = self.block1_seq1(x)

        x = self.block1_seq2(x)

        x = self.block1_seq3(x)

        x = self.block1_seq4(x)
        x = self.block2_seq4(x)

        x = self.block1_seq5(x)
        x = self.block2_seq5(x)
        x = self.block2_seq5_2(x)

        x = self.block1_seq6(x)
        x = self.block2_seq6(x)
        x = self.block2_seq6_2(x)

        output = self.module5(x)

        return output


class Block1(Model):
    def __init__(self, conf, rate, nc):
        """
        Parameters
        ----------
        conf: Dictionary
            Configuration dictionary.
        """

        # Initialize.
        self.conf = conf
        self.raw_data_path = self.conf['raw_data_path']
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']

        super(Block1, self).__init__()

        # Design layers.
        self.module1 = Module1(conf, rate, nc)
        self.module2 = Module2(conf, rate, int(nc * 1.5))
        self.module3 = Module3(conf, int(nc * 1.5))
        self.module4 = Module4(conf, rate, int(nc * 1.5))
        self.module7 = Module7(conf, rate, int(nc * 1.5))

    def call(self, input_tensor):
        x2 = self.module1(input_tensor)
        x3 = self.module2(x2)
        x4 = self.module3(x2)
        x5 = self.module4([x3, x4])
        x6 = self.module7([x3, x5])

        return x6


class Block2(Model):
    def __init__(self, conf, rate, nc):
        """
        Parameters
        ----------
        conf: Dictionary
            Configuration dictionary.
        """

        # Initialize.
        self.conf = conf
        self.raw_data_path = self.conf['raw_data_path']
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']

        super(Block2, self).__init__()

        # Design layers.
        self.module2 = Module2(conf, rate, nc)
        self.module3 = Module3(conf, nc)
        self.module4 = Module4(conf, rate, nc)
        self.module7 = Module7(conf, rate, nc)

    def call(self, input_tensor):
        x2 = self.module2(input_tensor)
        x3 = self.module3(x2)
        x4 = self.module4([x2, x3])
        x5 = self.module7([x3, x4])

        return x5


class Block3(Model):
    def __init__(self, conf, rate, nc):
        """
        Parameters
        ----------
        conf: Dictionary
            Configuration dictionary.
        """

        # Initialize.
        self.conf = conf
        self.raw_data_path = self.conf['raw_data_path']
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']

        super(Block3, self).__init__()

        # Design layers.
        self.module6 = Module6(conf, nc)
        self.module2 = Module2(conf, rate, nc)
        self.module3 = Module3(conf, nc)
        self.module4 = Module4(conf, rate, nc)
        self.module7 = Module7(conf, rate, nc)

    def call(self, input_tensor):
        x2 = self.module6(input_tensor)
        x3 = self.module2(x2)
        x4 = self.module3(x2)
        x5 = self.module4([x3, x4])
        x6 = self.module7([x3, x5])

        return x6


class Module1(Model):
    def __init__(self, conf, rate, nc):
        """
        Parameters
        ----------
        conf: Dictionary
            Configuration dictionary.
        """

        # Initialize.
        self.conf = conf
        self.raw_data_path = self.conf['raw_data_path']
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']

        super(Module1, self).__init__()

        # Design layers.
        self.sep_conv3d_1 = SeparableConv3D(nc
                             , kernel_size=3
                             , depth_multiplier=1
                             , dilation_rate=(rate[0] * self.nn_arch['conv_rate_multiplier']
                                              , rate[1] * self.nn_arch['conv_rate_multiplier']
                                              , rate[2] * self.nn_arch['conv_rate_multiplier'])
                             , padding='same'
                             , use_bias=False
                             , kernel_initializer=initializers.TruncatedNormal())
        self.bn_1 = BatchNormalization(momentum=self.hps['bn_momentum'], scale=self.hps['bn_scale'])
        self.act_1 = Activation('relu')

        self.conv3d_2 = Conv3D(int(nc * 1.5)
                    , kernel_size=1
                    , strides=2
                    , padding='same'
                    , use_bias=False
                    , kernel_initializer=initializers.TruncatedNormal()
                    , kernel_regularizer=regularizers.l2(self.hps['weight_decay']))
        self.bn_2 = BatchNormalization(momentum=self.hps['bn_momentum'], scale=self.hps['bn_scale'])
        self.act_2 = Activation('relu')

    def call(self, input_tensor):
        x = self.sep_conv3d_1(input_tensor)
        x = self.bn_1(x)
        x = self.act_1(x)

        x = self.conv3d_2(x)
        x = self.bn_2(x)
        x = self.act_2(x)

        return x


class Module2(Model):
    def __init__(self, conf, rate, nc):
        """
        Parameters
        ----------
        conf: Dictionary
            Configuration dictionary.
        """

        # Initialize.
        self.conf = conf
        self.raw_data_path = self.conf['raw_data_path']
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']

        super(Module2, self).__init__()

        # Design layers.
        # Split separable conv3d.
        self.sep_conv3d_1 = SeparableConv3D(nc
                             , kernel_size=3
                             , depth_multiplier=1
                             , dilation_rate=(rate[0] * self.nn_arch['conv_rate_multiplier']
                                              , rate[1] * self.nn_arch['conv_rate_multiplier']
                                              , rate[2] * self.nn_arch['conv_rate_multiplier'])
                             , padding='same'
                             , use_bias=False
                             , kernel_initializer=initializers.TruncatedNormal())
        self.bn_1 = BatchNormalization(momentum=self.hps['bn_momentum'], scale=self.hps['bn_scale'])
        self.act_1 = Activation('relu')

        self.conv3d_2 = Conv3D(np.maximum(1, int(nc / 2))
                    , kernel_size=3
                    , strides=2
                    , padding='valid'
                    , use_bias=False
                    , kernel_initializer=initializers.TruncatedNormal()
                    , kernel_regularizer=regularizers.l2(self.hps['weight_decay']))
        self.bn_2 = BatchNormalization(momentum=self.hps['bn_momentum'], scale=self.hps['bn_scale'])
        self.act_2 = Activation('relu')

        self.sep_conv3d_3 = SeparableConv3D(nc
                             , kernel_size=3
                             , depth_multiplier=1
                             , dilation_rate=(rate[0] * self.nn_arch['conv_rate_multiplier']
                                              , rate[1] * self.nn_arch['conv_rate_multiplier']
                                              , rate[2] * self.nn_arch['conv_rate_multiplier'])
                             , padding='same'
                             , use_bias=False
                             , kernel_initializer=initializers.TruncatedNormal())
        self.bn_3 = BatchNormalization(momentum=self.hps['bn_momentum'], scale=self.hps['bn_scale'])
        self.act_3 = Activation('relu')

    def call(self, input_tensor):
        x = self.sep_conv3d_1(input_tensor)
        x = self.bn_1(x)
        x = self.act_1(x)

        x = self.conv3d_2(x)
        x = self.bn_2(x)
        x = self.act_2(x)

        x = self.sep_conv3d_3(x)
        x = self.bn_3(x)
        x = self.act_3(x)

        return x


class Module3(Model):
    def __init__(self, conf, nc):
        """
        Parameters
        ----------
        conf: Dictionary
            Configuration dictionary.
        """

        # Initialize.
        self.conf = conf
        self.raw_data_path = self.conf['raw_data_path']
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']

        super(Module3, self).__init__()

        # Design layers.
        self.global_avg_pool3d_1 = GlobalAveragePooling3D() # data_format?
        self.reshape_1 = Reshape((1, 1, 1, nc))
        self.conv3d_1 = Conv3D(np.maximum(1, int(nc / 2))
                    , kernel_size=1
                    , strides=1
                    , padding='same'
                    , use_bias=False
                    , kernel_initializer=initializers.TruncatedNormal()
                    , kernel_regularizer=regularizers.l2(self.hps['weight_decay']))

        self.conv3d_2 = Conv3D(nc
                    , kernel_size=1
                    , strides=1
                    , padding='same'
                    , use_bias=False
                    , kernel_initializer=initializers.TruncatedNormal()
                    , kernel_regularizer=regularizers.l2(self.hps['weight_decay']))

    def call(self, input_tensor):
        x = self.global_avg_pool3d_1(input_tensor)
        x = self.reshape_1(x)
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)

        return x


class Module4(Model):
    def __init__(self, conf, rate, nc):
        """
        Parameters
        ----------
        conf: Dictionary
            Configuration dictionary.
        """

        # Initialize.
        self.conf = conf
        self.raw_data_path = self.conf['raw_data_path']
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']

        super(Module4, self).__init__()

        # Design layers.
        self.multiply_1 = Multiply()
        self.sep_conv3d_1 = SeparableConv3D(nc
                                            , kernel_size=3
                                            , depth_multiplier=1
                                            , dilation_rate=(rate[0] * self.nn_arch['conv_rate_multiplier']
                                                             , rate[1] * self.nn_arch['conv_rate_multiplier']
                                                             , rate[2] * self.nn_arch['conv_rate_multiplier'])
                                            , padding='same'
                                            , use_bias=False
                                            , kernel_initializer=initializers.TruncatedNormal())
        self.bn_1 = BatchNormalization(momentum=self.hps['bn_momentum'], scale=self.hps['bn_scale'])
        self.act_1 = Activation('relu')
        self.dropout_1 = Dropout(rate=self.nn_arch['dropout_rate'])

    def call(self, inputs):
        # Check exception.
        x = inputs
        if isinstance(x, list) != True or len(x) != 2:
            raise ValueError('Input must be a list of two tensors.')

        x = self.multiply_1([x[0], x[1]])
        x = self.sep_conv3d_1(x)
        x = self.bn_1(x)
        x = self.act_1(x)
        x = self.dropout_1(x)

        return x


class Module5(Model):
    def __init__(self, conf, nc):
        """
        Parameters
        ----------
        conf: Dictionary
            Configuration dictionary.
        """

        # Initialize.
        self.conf = conf
        self.raw_data_path = self.conf['raw_data_path']
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']

        super(Module5, self).__init__()

        # Design layers.
        self.conv3d_1 = Conv3D(nc
                               , kernel_size=3
                               , strides=1
                               , padding='same'
                               , use_bias=False
                               , kernel_initializer=initializers.TruncatedNormal()
                               , kernel_regularizer=regularizers.l2(self.hps['weight_decay']))

    def call(self, input_tensor):
        x = self.conv3d_1(input_tensor)
        return x


class Module6(Model):
    def __init__(self, conf, nc):
        """
        Parameters
        ----------
        conf: Dictionary
            Configuration dictionary.
        """

        # Initialize.
        self.conf = conf
        self.raw_data_path = self.conf['raw_data_path']
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']

        super(Module6, self).__init__()

        # Design layers.
        self.upsampling3d_1 = UpSampling3D()
        self.conv3d_1 = Conv3D(nc
                               , kernel_size=3
                               , strides=1
                               , padding='same'
                               , use_bias=False
                               , kernel_initializer=initializers.TruncatedNormal()
                               , kernel_regularizer=regularizers.l2(self.hps['weight_decay']))
        self.bn_1 = BatchNormalization(momentum=self.hps['bn_momentum'], scale=self.hps['bn_scale'])
        self.act_1 = Activation('relu')

    def call(self, input_tensor):
        x = self.upsampling3d_1(input_tensor)
        x = self.conv3d_1(x)
        x = self.bn_1(x)
        x = self.act_1(x)

        return x

class Module7(Model):
    def __init__(self, conf, rate, nc):
        """
        Parameters
        ----------
        conf: Dictionary
            Configuration dictionary.
        """

        # Initialize.
        self.conf = conf
        self.raw_data_path = self.conf['raw_data_path']
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']

        super(Module7, self).__init__()

        # Design layers.
        self.add_1 = Add()
        self.conv3d_1 = Conv3D(nc
                            , kernel_size=3
                            , dilation_rate=(rate[0] * self.nn_arch['conv_rate_multiplier']
                                             , rate[1] * self.nn_arch['conv_rate_multiplier']
                                             , rate[2] * self.nn_arch['conv_rate_multiplier'])
                            , padding='same'
                            , use_bias=False
                            , kernel_initializer=initializers.TruncatedNormal())
        self.bn_1 = BatchNormalization(momentum=self.hps['bn_momentum'], scale=self.hps['bn_scale'])
        self.act_1 = Activation('relu')

    def call(self, inputs):
        # Check exception.
        x = inputs
        if isinstance(x, list) != True or len(x) != 2:
            raise ValueError('Input must be a list of two tensors.')

        x = self.add_1([x[0], x[1]])
        x = self.conv3d_1(x)
        x = self.bn_1(x)
        x = self.act_1(x)

        return x