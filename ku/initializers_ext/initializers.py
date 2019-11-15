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

from tensorflow_core.python.ops.init_ops import VarianceScaling

def he_normal(seed=None, scale=2.):
    """He normal initializer.
    It draws samples from a truncated normal distribution centered on 0
    with standard deviation (after truncation) given by
    `stddev = sqrt(2 / fan_in)` where `fan_in` is the number of
    input units in the weight tensor.
      
    Parameters
    ----------
    seed: A Python integer. Used to seed the random generator.
    
    Returns
    -------
    An initializer.
    
    References
    ----------
        [He et al., 2015]
        (https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)
        # pylint: disable=line-too-long
        ([pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf))
    """
    return VarianceScaling(
        scale=scale, mode="fan_in", distribution="truncated_normal", seed=seed)