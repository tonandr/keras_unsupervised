# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.framework import ops, smart_cond
from tensorflow.python.ops import math_ops, array_ops
import tensorflow.keras.backend as K
from tensorflow.python.keras.utils import losses_utils

EPSILON = 1e-8

class CategoricalCrossentropyWithLabelGT(LossFunctionWrapper):
    def __init__(self
                 , num_classes=2
                 , from_logits=False
                 , label_smoothing=0
                 , reduction=losses_utils.ReductionV2.AUTO
                 , name='categorical_corssentropy_with_label_gt'):
        super(CategoricalCrossentropyWithLabelGT, self).__init__(
            categorical_corssentropy_with_label_gt
            , name=name
            , reduction=reduction
            , num_classes=num_classes
            , from_logits=from_logits
            , label_smoothing=label_smoothing) 

class GenDiscRegularLoss1(LossFunctionWrapper):
    def __init__(self
                 , reduction=losses_utils.ReductionV2.AUTO
                 , name='gen_disc_regular_loss1'):
        super(GenDiscRegularLoss1, self).__init__(
            gen_disc_regular_loss1, name=name, reduction=reduction)

class GenDiscRegularLoss2(LossFunctionWrapper):
    def __init__(self
                 , reduction=losses_utils.ReductionV2.AUTO
                 , name='gen_disc_regular_loss2'):
        super(GenDiscRegularLoss2, self).__init__(
            gen_disc_regular_loss2, name=name, reduction=reduction)

class DiscExtRegularLoss1(LossFunctionWrapper):
    def __init__(self
                 , reduction=losses_utils.ReductionV2.AUTO
                 , name='disc_ext_regular_loss1'):
        super(DiscExtRegularLoss1, self).__init__(
            disc_ext_regular_loss1, name=name, reduction=reduction)

class DiscExtRegularLoss2(LossFunctionWrapper):
    def __init__(self
                 , reduction=losses_utils.ReductionV2.AUTO
                 , name='disc_ext_regular_loss2'):
        super(DiscExtRegularLoss2, self).__init__(
            disc_ext_regular_loss2, name=name, reduction=reduction)

class GenDiscWGANLoss(LossFunctionWrapper):
    def __init__(self
                 , reduction=losses_utils.ReductionV2.AUTO
                 , name='gen_disc_wgan_loss'):
        super(GenDiscWGANLoss, self).__init__(
            gen_disc_wgan_loss, name=name, reduction=reduction)

class DiscExtWGANLoss(LossFunctionWrapper):
    def __init__(self
                 , reduction=losses_utils.ReductionV2.AUTO
                 , name='disc_ext_wgan_loss'):
        super(DiscExtWGANLoss, self).__init__(
            disc_ext_wgan_loss, name=name, reduction=reduction)

class DiscExtWGANGPLoss(LossFunctionWrapper):
    def __init__(self
                 , reduction=losses_utils.ReductionV2.AUTO
                 , name='disc_ext_wgan_gp_loss'
                 , input_variables=None
                 , wgan_lambda=10.0
                 , wgan_target=1.0):
        super(DiscExtWGANGPLoss, self).__init__(
            disc_ext_wgan_gp_loss
            , name=name
            , reduction=reduction
            , input_variables=input_variables
            , wgan_lambda = wgan_lambda
            , wgan_target = wgan_target)

class SoftPlusNonSatLoss(LossFunctionWrapper):
    def __init__(self
                 , reduction=losses_utils.ReductionV2.AUTO
                 , name='softplus_non_sat_loss'):
        super(SoftPlusNonSatLoss, self).__init__(
            softplus_non_sat_loss, name=name, reduction=reduction)

class SoftPlusLoss(LossFunctionWrapper):
    def __init__(self
                 , reduction=losses_utils.ReductionV2.AUTO
                 , name='softplus_loss'):
        super(SoftPlusLoss, self).__init__(
            softplus_loss, name=name, reduction=reduction)

class RPenaltyLoss(LossFunctionWrapper):
    def __init__(self
                 , reduction=losses_utils.ReductionV2.AUTO
                 , name='r_penalty_loss'
                 , model=None
                 , input_variable_orders=None
                 , r_gamma=10.0
                 , wgan_target=1.0):
        super(RPenaltyLoss, self).__init__(
            r_penalty_loss
            , name=name
            , reduction=reduction
            , model=model
            , input_variable_orders=input_variable_orders
            , r_gamma = r_gamma)

class SoftPlusNonSatRPenaltyLoss(LossFunctionWrapper):
    def __init__(self
                 , reduction=losses_utils.ReductionV2.AUTO
                 , name='softplus_non_sat_r_penalty_loss'
                 , input_variables=None
                 , r_gamma=10.0
                 , wgan_target=1.0):
        super(SoftPlusNonSatRPenaltyLoss, self).__init__(
            softplus_non_sat_r_penalty_loss
            , name=name
            , reduction=reduction
            , input_variables=input_variables
            , r_gamma = r_gamma)

def categorical_corssentropy_with_label_gt(y_true, y_pred, num_classes=2, from_logits=False, label_smoothing=0):
    y_true = K.one_hot(math_ops.cast(y_true, 'int32'), num_classes)
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    label_smoothing = ops.convert_to_tensor(label_smoothing, dtype=K.floatx())
    
    def _smooth_labels():
        num_classes = math_ops.cast(array_ops.shape(y_true)[1], y_pred.dtype)
        return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)
    
    y_true = smart_cond.smart_cond(label_smoothing,
                                   _smooth_labels, lambda: y_true)
    return K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)

def gen_disc_regular_loss1(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return K.mean(K.log(1.0 - y_pred + EPSILON), axis=-1)# Axis?

def gen_disc_regular_loss2(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return K.mean(K.log(y_pred + EPSILON), axis=-1)

def disc_ext_regular_loss1(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return K.mean(K.log(y_pred + EPSILON), axis=-1)

def disc_ext_regular_loss2(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return K.mean(K.log(1.0 - y_pred + EPSILON), axis=-1)

def gen_disc_wgan_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return K.mean(y_pred, axis=-1)

def disc_ext_wgan_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return K.mean(y_pred, axis=-1)

def disc_ext_wgan_gp_loss(y_true
                          , y_pred
                          , input_variables=None
                          , wgan_lambda=10.0
                          , wgan_target=1.0):
    if input_variables is None:
        raise ValueError('input_variables must be assigned.')
    global tape #?
    assert tf.executing_eagerly() and 'tape' in dir() and isinstance(tape, tf.GradientTape) and tape._persistent
    
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)    
    grads = tape.gradient(y_pred, input_variables)
    norm = K.sqrt(K.sum(K.square(grads), axis=[1, 2, 3])) #?
    return (wgan_lambda / (wgan_target ** 2)) * K.square(norm - wgan_target) #?

def softplus_non_sat_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return K.softplus(-1.0 * y_pred) #?

def softplus_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return K.softplus(y_pred)

def r_penalty_loss(y_true, y_pred, model, input_variable_orders=None, r_gamma=10.0): #?
    if model is None or model.assigned_inputs is None or model.tape_handler is None or input_variable_orders is None:
        raise ValueError('model and model.assigned_inputs and model.tape_handler and input_variable_orders must be assigned.')
    assert tf.executing_eagerly() and isinstance(model.tape_handler, tf.GradientTape) and model.tape_handler._persistent

    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    inputs = [model.assigned_inputs[k] for k in input_variable_orders]       
    grads = model.tape_handler.gradient(K.sum(y_pred, axis=-1), inputs)
    r_penalty = K.sum(K.square(grads), axis=[1, 2, 3])
    return r_gamma * 0.5 * r_penalty
    
def softplus_non_sat_r_penalty_loss(y_true, y_pred, input_variables, r_gamma = 10.0): #?
    if input_variables is None:
        raise ValueError('input_variables must be assigned.')
    global tape #?
    assert tf.executing_eagerly() and 'tape' in dir() and isinstance(tape, tf.GradientTape) and tape._persistent

    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)     
    grads = K.gradients(K.sum(y_pred, axis=-1), input_variables)[0]
    r_penalty = K.sum(K.square(grads), axis=[1, 2, 3])
    return K.softplus(-1.0 * y_pred) + r_gamma * 0.5 * r_penalty
