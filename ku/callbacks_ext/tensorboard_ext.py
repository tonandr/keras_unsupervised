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
    def __init__(self
               , log_dir='logs'
               , histogram_freq=0
               , write_graph=True
               , write_images=False
               , update_freq='epoch'
               , profile_batch=2
               , embeddings_freq=0
               , embeddings_metadata=None
               , **kwargs):
        super(TensorBoardExt, self).__init__(log_dir=log_dir
               , histogram_freq=histogram_freq
               , write_graph=write_graph
               , write_images=write_images
               , update_freq=update_freq
               , profile_batch=profile_batch
               , embeddings_freq=embeddings_freq
               , embeddings_metadata=embeddings_metadata
               , **kwargs)
    
    def on_batch_end(self, batch, logs=None):
        """Writes scalar summaries for metrics on every training batch.
        Performs profiling if current batch is in profiler_batches.
        
        Parameters
        ----------
            batch: Integer
                Index of batch within the current epoch.
            logs: Dict
                 Metric results for this batch.
        """
                
        # Don't output batch_size and batch number as TensorBoard summaries
        logs = logs or {}
        self._samples_seen += logs.get('size', 1)
        samples_seen_since = self._samples_seen - self._samples_seen_at_last_write
        if self.update_freq != 'epoch' and samples_seen_since >= self.update_freq:
            self._log_metrics(logs, prefix='batch_', step=self._total_batches_seen)
            self._log_weights('batch', step=self._total_batches_seen)
            self._samples_seen_at_last_write = self._samples_seen
        self._total_batches_seen += 1
        if self._is_tracing:
            self._log_trace()
        elif (not self._is_tracing and
                self._total_batches_seen == self._profile_batch - 1):
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
        step = epoch if self.update_freq == 'epoch' else self._samples_seen
        self._log_metrics(logs, prefix='epoch_', step=step)
    
        if self.histogram_freq and epoch % self.histogram_freq == 0: #?
            self._log_weights('epoch', epoch)
    
        if self.embeddings_freq and epoch % self.embeddings_freq == 0:
            self._log_embeddings(epoch)          