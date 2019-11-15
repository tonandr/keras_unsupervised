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

from tensorflow.keras.metrics import MeanIoU
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops, array_ops, confusion_matrix
import tensorflow.keras.backend as K

class MeanIoUExt(MeanIoU):
    """Calculate the mean IoU for one hot truth and prediction vectors."""
    def __init__(self, num_classes, accum_enable=True, name=None, dtype=None):
        super(MeanIoUExt, self).__init__(num_classes, name=name, dtype=dtype)
        self.accum_enable = accum_enable
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulated the confusion matrix statistics with one hot truth and prediction data.
        
        Parameters
        ----------
        y_true: Tensor or numpy array. 
            One hot ground truth vectors.
        y_pred: Tensor or numpy array.
            One hot predicted vectors.
        sample_weight: Tensor.
            Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.
            
        Returns
        -------
        Update operator.
            Operator
        """
        # Convert one hot vectors and labels.
        y_pred = K.argmax(y_pred)
               
        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)
                
        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = array_ops.reshape(y_pred, [-1])
    
        if y_true.shape.ndims > 1:
            y_true = array_ops.reshape(y_true, [-1])
    
        if sample_weight is not None and sample_weight.shape.ndims > 1:
            sample_weight = array_ops.reshape(sample_weight, [-1])

        # Accumulate the prediction to current confusion matrix.
        current_cm = confusion_matrix.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes,
            weights=sample_weight,
            dtype=dtypes.float64)
        return self.total_cm.assign_add(current_cm) if self.accum_enable \
            else self.total_cm.assign(current_cm)
        