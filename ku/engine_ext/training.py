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

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import training_utils

from . import training_eager
from . import training_v2_utils

class ModelExt(Model):
    # Constants.
    PROGRESSIVE_MODE_FORWARD = 0
    PROGRESSIVE_MODE_BACKWARD = 1
    
    def __init__(self, *args, **kwargs):
        super(ModelExt, self).__init__(*args, **kwargs)
        self.tape_handler = None
      
    def train_on_batch(self
                       , x
                       , y=None
                       , sample_weight=None
                       , class_weight=None
                       , reset_metrics=True):
        """Runs a single gradient update on a single batch of data.
        
        Parameters
        ----------
            x: Input data. It could be:
                - A Numpy array (or array-like), or a list of arrays
                    (in case the model has multiple inputs).
                - A TensorFlow tensor, or a list of tensors
                    (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding array/tensors,
                    if the model has named inputs.
                - A `tf.data` dataset.
            y: Target data. Like the input data `x`, it could be either Numpy
                array(s) or TensorFlow tensor(s). It should be consistent with `x`
                (you cannot have Numpy inputs and tensor targets, or inversely). If
                `x` is a dataset, `y` should not be specified
                (since targets will be obtained from the iterator).
            sample_weight: Optional array of the same length as x, containing
                weights to apply to the model's loss for each sample. In the case of
                temporal data, you can pass a 2D array with shape (samples,
                sequence_length), to apply a different weight to every timestep of
                every sample. In this case you should make sure to specify
                sample_weight_mode="temporal" in compile(). This argument is not
                supported when `x` is a dataset.
            class_weight: Optional dictionary mapping class indices (integers) to a
                weight (float) to apply to the model's loss for the samples from this
                class during training. This can be useful to tell the model to "pay
                more attention" to samples from an under-represented class.
            reset_metrics: If `True`, the metrics returned will be only for this
                batch. If `False`, the metrics will be statefully accumulated across
                batches.
        Returns
        -------
            Scalar training loss
            (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        Raises
        ------
            ValueError: In case of invalid user-provided arguments.
        """
        self._assert_compile_was_called()
        self._check_call_args('train_on_batch')
        if self._experimental_run_tf_function:
            outputs = training_v2_utils.train_on_batch(
                self, x, y=y, sample_weight=sample_weight,
                class_weight=class_weight, reset_metrics=reset_metrics)
            outputs = (outputs['total_loss'] + outputs['output_losses'] +
                       outputs['metrics'])
            outputs = [
                training_v2_utils._non_none_constant_value(v) for v in outputs]
            if len(outputs) == 1:
                outputs = outputs[0]
            return outputs
    
        # If at this point we are in the replica context, then it is okay to execute
        # the Eager code path.  The expected way to get here is to call `fit` that
        # calls `train_on_batch` on each replica.
        if (self._distribution_strategy and
            distribution_strategy_context.in_cross_replica_context()):
            raise NotImplementedError('`train_on_batch` is not supported for models '
                                      'distributed with tf.distribute.Strategy.')
        # Validate and standardize user data.
        x, y, sample_weights = self._standardize_user_data(
            x, y, sample_weight=sample_weight, class_weight=class_weight,
            extract_tensors_from_dataset=True)
    
        # If `self._distribution_strategy` is True, then we are in a replica context
        # at this point because of the check above.  `train_on_batch` is being run
        # for each replica by `self._distribution_strategy` and the same code path
        # as Eager is expected to be taken.
        if self.run_eagerly or self._distribution_strategy:
            output_dict = training_eager.train_on_batch(self
                , x
                , y
                , sample_weights=sample_weights,
                output_loss_metrics=self._output_loss_metrics)
            outputs = (output_dict['total_loss'] + output_dict['output_losses']
                       + output_dict['metrics'])
            outputs = [training_v2_utils._non_none_constant_value(v) for v in outputs]
        else:
            x = training_utils.ModelInputs(x).as_list()
            ins = x + (y or []) + (sample_weights or [])
    
            if not isinstance(K.symbolic_learning_phase(), int):
                ins += [True]  # Add learning phase value.
    
            self._update_sample_weight_modes(sample_weights=sample_weights)
            self._make_train_function()
            outputs = self.train_function(ins)
    
        if reset_metrics:
            self.reset_metrics()
    
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def glue_layers(self, gluing_layers, glued_layer_names):
        """Glue layers to the model.
        
        Parameters
        ----------
            gluing_layers: List or tuple.
                Gluing layers.
            glued_layer_names: List or tuple.
                Glued layer names between beginning and ending.
        """
        
        # Check exception.
        layer_names = [layer.name for layer in self.layers] 
        
        assert len(glued_layer_names) == 2 and (glued_layer_names[0] or glued_layer_names[0]) #? 
        if glued_layer_names[0]: #?
            assert glued_layer_names[0] in layer_names
        if glued_layer_names[1]:
            assert glued_layer_names[1] in layer_names
        if not glued_layer_names[0] and glued_layer_names[1]:
            assert isinstance(gluing_layers[0], tf.Tensor) and len(gluing_layers) >= 2 #?
            
        # Glue layers according to glued layers. 
        if glued_layer_names[0] and not glued_layer_names[1]:
            s_idx = layer_names.index(glued_layer_names[0])
            upper_layer_name_idxes = range(s_idx + 1)
            
            # Inputs.
            inputs =[[Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in model_input] \
                    for model_input in self.inputs] 
            
            # Glued upper layers.
            x = inputs #?
            for i, layer_name_idx in enumerate(upper_layer_name_idxes):
                x = self.get_layer(layer_names[layer_name_idx])(x)
            
            # Gluing layers.
            for layer in gluing_layers:
                x = layer(x) # Exception?
            
            return Model(inputs, x) #?
        elif glued_layer_names[0] and glued_layer_names[1]:
            s_idx = layer_names.index(glued_layer_names[0])
            e_idx = layer_names.index(glued_layer_names[1])
            
            # Inputs.
            inputs =[[Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in model_input] \
                    for model_input in self.inputs] 
            
            # Glued upper layers.
            x = inputs #?
            for idx in range(s_idx):
                x = self.get_layer(layer_names[idx])(x)
            
            # Gluing layers.
            for layer in gluing_layers:
                x = layer(x) # Exception?
            
            # Glued lower layers.
            for idx in range(e_idx, len(layer_names)):
                x = self.get_layer(layer_names[idx])(x)            
            
            return Model(inputs, x) #?
        else:
            e_idx = layer_names.index(glued_layer_names[1])
            lower_layer_name_idxes = range(e_idx, len(layer_names))
            
            # Inputs.
            inputs = gluing_layers[0]

            # Gluing layers.
            x = inputs #?
            for layer in gluing_layers[1:]:
                x = layer(x) # Exception?
            
            # Glued lower layers.
            for i, layer_name_idx in enumerate(lower_layer_name_idxes):
                x = self.get_layer(layer_names[layer_name_idx])(x)
            
            return Model(inputs, x) #?
    
    def create_prog_model(self, prog_mode, prog_depth, fixed_layer_names, compile_f=True):
        """Create a progressive model for progressive learning.
        
        Parameters
        ----------
        prog_mode: Integer.
            Progressive mode: Forward (0), backward (1).
        prog_depth: Integer.
            Depth of a progressive model.
        fixed_layer_names: List or tuple.
            Layer names not learned progressively.
        compile_f: Boolean.
            Progressive model compiling flag (default: True).
        """
        
        # Check exception.
        layer_names = [layer.name for layer in self.layers]
        total_depth = len(layer_names) 
        
        assert prog_mode in [self.PROGRESSIVE_MODE_FORWARD, self.PROGRESSIVE_MODE_BACKWARD] \
            and len(fixed_layer_names) == 2
        if prog_mode == self.PROGRESSIVE_MODE_FORWARD:
            assert fixed_layer_names[0]
            assert fixed_layer_names[0] in layer_names
            fixed_s_layer_depth = layer_names.index(fixed_layer_names[0])
            assert fixed_s_layer_depth < prog_depth
            
            if fixed_layer_names[1]: 
                assert fixed_layer_names[1] in layer_names
                fixed_e_layer_depth = layer_names.index(fixed_layer_names[1])
                assert fixed_e_layer_depth > prog_depth
            else:
                fixed_e_layer_depth = -1
        if prog_mode == self.PROGRESSIVE_MODE_BACKWARD:
            assert not fixed_layer_names[0] and fixed_layer_names[1]
            assert fixed_layer_names[1] in layer_names
            fixed_e_layer_depth = layer_names.index(fixed_layer_names[1])
            assert fixed_e_layer_depth > prog_depth
            fixed_s_layer_depth = -1
        
        # Make a progressive model.
        if prog_mode == self.PROGRESSIVE_MODE_FORWARD:
            # Inputs.
            inputs =[[Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in model_input] \
                    for model_input in self.inputs]
            
            # Initial layers.
            x = inputs
            for idx in range(fixed_s_layer_depth + 1):
                x = self.get_layer(layer_names[idx])(x) # Input layer?
            
            # Middle layers.
            for idx in range(fixed_s_layer_depth + 1, prog_depth + 1):
                x = self.get_layer(layer_names[idx])(x)
            
            # Final layers.
            if fixed_e_layer_depth != -1:
                for idx in range(fixed_e_layer_depth, total_depth):
                    x = self.get_layer(layer_names[idx])(x)
            
            outputs = x
            prog_model = ModelExt(inputs, outputs, name='forward_prog_model') #?
            
            if compile_f:
                assert self._is_compiled
                prog_model.compile(optimizer=self.optimizer 
                                    , loss=self.losses
                                    , loss_weights=self.loss_weights
                                    , run_eagerly=True) #?         
        else:
            # Inputs.
            inputs = Input(K.int_shape(self.get_layer(layer_names[prog_depth]).input)[1:]) # Multiple inputs?
            
            # Initial and middle layers.
            x = inputs
            for idx in range(prog_depth, fixed_e_layer_depth):
                x = self.get_layer(layer_names[idx])(x) # Input layer?
            
            # Final layers.
            for idx in range(fixed_e_layer_depth, total_depth):
                x = self.get_layer(layer_names[idx])(x)
                        
            outputs = x
            prog_model = ModelExt(inputs, outputs, name='backward_prog_model') #?
            
            if compile_f:
                assert self._is_compiled
                prog_model.compile(optimizer=self.optimizer 
                                    , loss=self.losses
                                    , loss_weights=self.loss_weights
                                    , run_eagerly=True) #?
        
        return prog_model
    
    def create_inner_prog_model(self, prog_mode, prog_depth, fixed_layer_names, compile_f=True):
        """Create a progressive model for progressive learning within the current model.
        
        Parameters
        ----------
        prog_mode: Integer.
            Progressive mode: Forward (0), backward (1).
        prog_depth: Integer.
            Depth of a progressive model.
        fixed_layer_names: List or tuple.
            Layer names not learned progressively.
        compile_f: Boolean.
            Progressive model compiling flag (default: True).
        """
        assert prog_mode in [self.PROGRESSIVE_MODE_FORWARD, self.PROGRESSIVE_MODE_BACKWARD] \
            and len(fixed_layer_names) == 2
            
        if prog_mode == self.PROGRESSIVE_MODE_FORWARD:
            self.forward_prog_model = self.create_prog_model(prog_mode
                                                             , prog_depth
                                                             , fixed_layer_names
                                                             , compile_f=compile_f)
        else:
            self.backward_prog_model = self.create_prog_model(prog_mode
                                                             , prog_depth
                                                             , fixed_layer_names
                                                             , compile_f=compile_f)
    
    @property
    def is_forward_prog_model(self):
        return hasattr(self, 'forward_prog_model')

    @property
    def is_backward_prog_model(self):
        return hasattr(self, 'forward_prog_model')
    
    @property
    def total_depth(self):
        return len(self.layers)
    
    def train_on_batch_forward_prog_model(self
                       , x
                       , y=None
                       , sample_weight=None
                       , class_weight=None
                       , reset_metrics=True):
        """Runs a single gradient update on a single batch of data for the forward progressive model.
        
        Parameters
        ----------
            x: Input data. It could be:
                - A Numpy array (or array-like), or a list of arrays
                    (in case the model has multiple inputs).
                - A TensorFlow tensor, or a list of tensors
                    (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding array/tensors,
                    if the model has named inputs.
                - A `tf.data` dataset.
            y: Target data. Like the input data `x`, it could be either Numpy
                array(s) or TensorFlow tensor(s). It should be consistent with `x`
                (you cannot have Numpy inputs and tensor targets, or inversely). If
                `x` is a dataset, `y` should not be specified
                (since targets will be obtained from the iterator).
            sample_weight: Optional array of the same length as x, containing
                weights to apply to the model's loss for each sample. In the case of
                temporal data, you can pass a 2D array with shape (samples,
                sequence_length), to apply a different weight to every timestep of
                every sample. In this case you should make sure to specify
                sample_weight_mode="temporal" in compile(). This argument is not
                supported when `x` is a dataset.
            class_weight: Optional dictionary mapping class indices (integers) to a
                weight (float) to apply to the model's loss for the samples from this
                class during training. This can be useful to tell the model to "pay
                more attention" to samples from an under-represented class.
            reset_metrics: If `True`, the metrics returned will be only for this
                batch. If `False`, the metrics will be statefully accumulated across
                batches.
        Returns
        -------
            Scalar training loss
            (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        Raises
        ------
            ValueError: In case of invalid user-provided arguments.
        """
        assert hasattr(self, 'forward_prog_model') and self.forward_prog_model._is_compiled
        return self.forward_prog_model.train_on_batch(x
                                               , y=y 
                                               , sample_weight=sample_weight
                                               , class_weight=class_weight
                                               , reset_metrics=reset_metrics)

    def train_on_batch_backward_prog_model(self
                       , x
                       , y=None
                       , sample_weight=None
                       , class_weight=None
                       , reset_metrics=True):
        """Runs a single gradient update on a single batch of data for the backward progressive model.
        
        Parameters
        ----------
            x: Input data. It could be:
                - A Numpy array (or array-like), or a list of arrays
                    (in case the model has multiple inputs).
                - A TensorFlow tensor, or a list of tensors
                    (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding array/tensors,
                    if the model has named inputs.
                - A `tf.data` dataset.
            y: Target data. Like the input data `x`, it could be either Numpy
                array(s) or TensorFlow tensor(s). It should be consistent with `x`
                (you cannot have Numpy inputs and tensor targets, or inversely). If
                `x` is a dataset, `y` should not be specified
                (since targets will be obtained from the iterator).
            sample_weight: Optional array of the same length as x, containing
                weights to apply to the model's loss for each sample. In the case of
                temporal data, you can pass a 2D array with shape (samples,
                sequence_length), to apply a different weight to every timestep of
                every sample. In this case you should make sure to specify
                sample_weight_mode="temporal" in compile(). This argument is not
                supported when `x` is a dataset.
            class_weight: Optional dictionary mapping class indices (integers) to a
                weight (float) to apply to the model's loss for the samples from this
                class during training. This can be useful to tell the model to "pay
                more attention" to samples from an under-represented class.
            reset_metrics: If `True`, the metrics returned will be only for this
                batch. If `False`, the metrics will be statefully accumulated across
                batches.
        Returns
        -------
            Scalar training loss
            (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        Raises
        ------
            ValueError: In case of invalid user-provided arguments.
        """
        assert hasattr(self, 'backward_prog_model') and self.backward_prog_model._is_compiled
        return self.backward_prog_model.train_on_batch(x
                                               , y=y 
                                               , sample_weight=sample_weight
                                               , class_weight=class_weight
                                               , reset_metrics=reset_metrics)
                                        