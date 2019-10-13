from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.estimator import inputs
tfd = tfp.distributions

def pad(tensor
    , paddings
    , mode='CONSTANT'
    , name=None
    , constant_values=0):
    """Add padding to tensor."""
    return tf.pad(tensor
                  , paddings
                  , mode=mode
                  , name=name
                  , constant_values=constant_values)

def transpose(a
    , perm=None
    , name='transpose'
    , conjugate=False):
    """Transpose tensor."""
    return tf.transpose(a
                        , perm=perm
                        , name=name
                        , conjugate=conjugate)

def multivariate_normal_diag(loc=None
                             , scale_diag=None
                             , scale_identity_multiplier=None
                             , validate_args=False
                             , allow_nan_stats=True
                             , name='MultivariateNormalDiag'):
    """Multivariate normal distribution with a diagonal covariance matrix."""
    
    # Check exception.
    return tfd.MultivariateNormalDiag(loc
                                      , scale_diag
                                      , scale_identity_multiplier
                                      , validate_args
                                      , allow_nan_stats
                                      , name)
    
def where(condition
          , x=None
          , y=None
          , name=None):
    return tf.where(condition, x, y, name)

def cond(pred
         , true_fn=None
         , false_fn=None
         , name=None):
    return tf.cond(pred, true_fn, false_fn, name)

def broadcast_to(input
                 , shape
                 , name=None):
    return tf.broadcast_to(input, shape, name)

def add_n(inputs
          , name=None):
    return tf.add_n(inputs, name)    
        