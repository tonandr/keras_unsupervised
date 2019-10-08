"""
COPYRIGHT

All contributions by François Chollet:
Copyright (c) 2015 - 2019, François Chollet.
All rights reserved.

All contributions by Google:
Copyright (c) 2015 - 2019, Google, Inc.
All rights reserved.

All contributions by Microsoft:
Copyright (c) 2017 - 2019, Microsoft, Inc.
All rights reserved.

All other contributions:
Copyright (c) 2015 - 2019, the respective contributors.
All rights reserved.

Each contributor holds copyright over their respective contributions.
The project versioning (Git) records all such contribution source information.

LICENSE

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.callbacks import TensorBoard
import tensorflow.keras.backend as K

class TensorBoardExt(TensorBoard): # tf 1.0?, license?
    def on_batch_end(self, batch, logs=None):
        if self.update_freq != 'epoch':
            self.samples_seen += logs['size']
            samples_seen_since = self.samples_seen - self.samples_seen_at_last_write
            if samples_seen_since >= self.update_freq:
                self._write_logs(logs, self.samples_seen)
                self.samples_seen_at_last_write = self.samples_seen
                
                # Log histogram frequency relevant results.        
                if not self.validation_data and self.histogram_freq: #?
                    raise ValueError("If printing histograms, validation_data must be "
                             "provided, and cannot be a generator.")
                
                if self.validation_data and self.histogram_freq:        
                    val_data = self.validation_data
                    tensors = (self.model.inputs +
                               self.model.targets +
                               self.model.sample_weights)
    
                    if self.model.uses_learning_phase:
                        tensors += [K.learning_phase()]
    
                    assert len(val_data) == len(tensors)
                    val_size = val_data[0].shape[0]
                    i = 0
                    while i < val_size:
                        step = min(self.batch_size, val_size - i)
                        if self.model.uses_learning_phase:
                            # do not slice the learning phase
                            batch_val = [x[i:i + step] for x in val_data[:-1]]
                            batch_val.append(val_data[-1])
                        else:
                            batch_val = [x[i:i + step] for x in val_data]
                        assert len(batch_val) == len(tensors)
                        feed_dict = dict(zip(tensors, batch_val))
                        result = self.sess.run([self.merged], feed_dict=feed_dict)
                        summary_str = result[0]
                        self.writer.add_summary(summary_str, batch) # batch?
                        i += self.batch_size        
