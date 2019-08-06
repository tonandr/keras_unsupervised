'''
Created on 2019. 7. 27.

@author: Inwoo Chung (gutomitai@gmail.com)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input
from keras import backend as K

EPSILON = 1e-8

def gen_disc_regular_loss1(y_true, y_pred):
    return K.mean(K.log(1.0 - y_pred + EPSILON), axis=-1)# Axis?

def gen_disc_regular_loss2(y_true, y_pred):
    return K.mean(K.log(y_pred + EPSILON), axis=-1)

def disc_ext_regular_loss1(y_true, y_pred):
    return K.mean(K.log(y_pred + EPSILON), axis=-1)

def disc_ext_regular_loss2(y_true, y_pred):
    return K.mean(K.log(1.0 - y_pred + EPSILON), axis=-1)

def gen_disc_wgan_loss(y_true, y_pred):
    return K.mean(y_pred, axis=-1)

def disc_ext_wgan_loss(y_true, y_pred):
    return K.mean(y_pred, axis=-1)

def disc_ext_wgan_gp_loss(input_variables, wgan_lambda = 10.0, wgan_target = 1.0):
    def _loss(y_true, y_pred):
        grads = K.gradients(y_pred, input_variables)[0] #?
        norm = K.sqrt(K.sum(K.square(grads), axis=[1, 2, 3])) #?
        return (wgan_lambda / (wgan_target ** 2)) * K.square(norm - wgan_target) #?
    return _loss

def softplus_non_sat_loss(y_true, y_pred):
    return K.softplus(-1.0 * y_pred) #?

def softplus_loss(y_true, y_pred):
    return K.softplus(y_pred)

def r_penalty_loss(input_variables, r_gamma = 10.0): #?
    def _loss(y_true, y_pred):
        grads = K.gradients(K.sum(y_pred, axis=-1), input_variables)[0]
        r_penalty = K.sum(K.square(grads), axis=[1, 2, 3])
        return r_gamma * 0.5 * r_penalty
    return _loss
    
def softplus_non_sat_r_penalty_loss(input_variables, r_gamma = 10.0): #?
    def _loss(y_true, y_pred):
        grads = K.gradients(K.sum(y_pred, axis=-1), input_variables)[0]
        r_penalty = K.sum(K.square(grads), axis=[1, 2, 3])
        return K.softplus(-1.0 * y_pred) + r_gamma * 0.5 * r_penalty
    return _loss