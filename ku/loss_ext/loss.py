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

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.framework import ops, smart_cond
from tensorflow.python.ops import math_ops, array_ops
from tensorflow.python.keras.utils import losses_utils

# Constants.
EPSILON = 1e-8

class CategoricalCrossentropyWithLabelGT(LossFunctionWrapper):
    def __init__(self
                 , num_classes=2
                 , from_logits=False
                 , label_smoothing=0
                 , reduction=losses_utils.ReductionV2.AUTO
                 , name='categorical_corssentropy_with_label_gt'):
        super(CategoricalCrossentropyWithLabelGT, self).__init__(categorical_corssentropy_with_label_gt
            , name=name
            , reduction=reduction
            , num_classes=num_classes
            , from_logits=from_logits
            , label_smoothing=label_smoothing) 

class WGANLoss(LossFunctionWrapper):
    def __init__(self
                 , reduction=losses_utils.ReductionV2.AUTO
                 , name='wgan_loss'):
        super(WGANLoss, self).__init__(
            wgan_loss, name=name, reduction=reduction)

class WGANGPLoss(LossFunctionWrapper):
    def __init__(self
                 , reduction=losses_utils.ReductionV2.AUTO
                 , name='wgan_gp_loss'
                 , model=None
                 , input_variable_orders=None
                 , wgan_lambda=10.0
                 , wgan_target=1.0):
        super(WGANGPLoss, self).__init__(wgan_gp_loss
            , name=name
            , reduction=reduction
            , model=model
            , input_variable_orders=input_variable_orders
            , wgan_lambda = wgan_lambda
            , wgan_target = wgan_target)

class SoftPlusInverseLoss(LossFunctionWrapper):
    def __init__(self
                 , reduction=losses_utils.ReductionV2.AUTO
                 , name='softplus_inverse_loss'):
        super(SoftPlusInverseLoss, self).__init__(
            softplus_inverse_loss, name=name, reduction=reduction)

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

def wgan_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return K.mean(y_pred, axis=-1)

def wgan_gp_loss(y_true
                , y_pred
                , model
                , input_variable_orders=None
                , wgan_lambda=10.0
                , wgan_target=1.0):
    if model is None or model.tape_handler is None or input_variable_orders is None:
        raise ValueError('model and model.assigned_inputs and model.tape_handler and input_variable_orders must be assigned.')
    assert tf.executing_eagerly() and isinstance(model.tape_handler, tf.GradientTape) and model.tape_handler._persistent
    
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    inputs = [model.assigned_inputs[k] for k in input_variable_orders]     
    grads = model.tape_handler.gradient(y_pred, inputs)
    norm = K.sqrt(K.sum(K.square(grads), axis=[1, 2, 3])) #?
    return (wgan_lambda / (wgan_target ** 2)) * K.square(norm - wgan_target) #?

def softplus_inverse_loss(y_true, y_pred):
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