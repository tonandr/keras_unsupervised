'''
Created on 2019. 5. 23.

@author: Inwoo Chung (gutomitai@gmail.com)
License: BSD 3 clause.

Revision:
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

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
         , strict=False
         , name=None
         , fn1=None
         , fn2=None):
    return tf.cond(pred, true_fn, false_fn, strict, name, fn1, fn2)

def broadcast_to(input
                 , shape
                 , name=None):
    return tf.broadcast_to(input, shape, name)    
        