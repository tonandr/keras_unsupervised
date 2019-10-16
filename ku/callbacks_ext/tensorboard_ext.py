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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2
import tensorflow.keras.backend as K

class TensorBoardExt(TensorBoard):
    def on_train_batch_end(self, batch, logs=None):
        """Writes scalar summaries for metrics on every training batch.
        Performs profiling if current batch is in profiler_batches.
        
        Parameters
        ----------
            batch: Integer
                Index of batch within the current epoch.
            logs: Dict
                 Metric results for this batch.
        """
        if self.update_freq == 'epoch' and self._profile_batch is None:
            return
        
        # Don't output batch_size and batch number as TensorBoard summaries
        logs = logs or {}
        train_batches = self._total_batches_seen[self._train_run_name]
        if self.update_freq != 'epoch' and batch % self.update_freq == 0:
            self._log_metrics(logs, prefix='batch_', step=train_batches)
            self._log_weights(prefix='batch', step=train_batches)
        
        self._increment_step(self._train_run_name)
        
        if context.executing_eagerly():
            if self._is_tracing:
                self._log_trace()
            elif (not self._is_tracing and
                math_ops.equal(train_batches, self._profile_batch - 1)):
                self._enable_trace()
        
    def _log_weights(self, prefix, step):
        """Logs the weights of the Model to TensorBoard.
        
        Parameters
        ----------
            prefix: String 
                The prefix to apply to the weight summary names.
            step: Int 
                The global step to use for TensorBoard.
        """
        writer = self._get_writer(self._train_run_name)
        with context.eager_mode(), \
              writer.as_default(), \
              summary_ops_v2.always_record_summaries():
            for layer in self.model.layers:
                for weight in layer.weights:
                    weight_name = prefix + weight.name.replace(':', '_') #?
                    with ops.init_scope():
                        weight = K.get_value(weight)
                    summary_ops_v2.histogram(weight_name, weight, step=step)
                    if self.write_images:
                        self._log_weight_as_image(weight, weight_name, step)
            writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        """Runs metrics and histogram summaries at epoch end."""
        self._log_metrics(logs, prefix='epoch_', step=epoch)
    
        if self.histogram_freq and epoch % self.histogram_freq == 0: #?
            self._log_weights(prefix='epoch', epoch)
    
        if self.embeddings_freq and epoch % self.embeddings_freq == 0:
            self._log_embeddings(epoch)          