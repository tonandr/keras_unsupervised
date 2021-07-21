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
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine.training import _minimize
from tensorflow.python.eager import backprop


class ModelExt(Model):
    '''Model extension'''

    # Constants.
    PROGRESSIVE_MODE_FORWARD = 0
    PROGRESSIVE_MODE_BACKWARD = 1
    
    def __init__(self, *args, **kwargs):
        super(ModelExt, self).__init__(*args, **kwargs)

    def train_step(self, data):
        """The logic for one training step.
        This method can be overridden to support custom training logic.
        This method is called by `Model.make_train_function`.
        This method should contain the mathemetical logic for one step of training.
        This typically includes the forward pass, loss calculation, backpropagation,
        and metric updates.
        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.
        Arguments:
          data: A nested structure of `Tensor`s.
        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided. These utilities will be exposed
        # publicly.
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        self.assigned_inputs = x
        with backprop.GradientTape(persistent=True) as tape: #?
            self.tape_handler = tape
            tape.watch(x)
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)
        # For custom training steps, users can just write:
        #   trainable_variables = self.trainable_variables
        #   gradients = tape.gradient(loss, trainable_variables)
        #   self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        # The _minimize call does a few extra steps unnecessary in most cases,
        # such as loss scaling and gradient clipping.
        _minimize(self.distribute_strategy, tape, self.optimizer, loss,
                  self.trainable_variables)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """The logic for one evaluation step.
        This method can be overridden to support custom evaluation logic.
        This method is called by `Model.make_test_function`.
        This function should contain the mathemetical logic for one step of
        evaluation.
        This typically includes the forward pass, loss calculation, and metrics
        updates.
        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_test_function`, which can also be overridden.
        Arguments:
          data: A nested structure of `Tensor`s.
        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned.
        """
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        # Updates stateful loss metrics.
        self.assigned_inputs = x
        with backprop.GradientTape(persistent=True) as tape: #?
            self.tape_handler = tape
            tape.watch(x)
            y_pred = self(x, training=False)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def glue_layers(self, gluing_layers, glued_layer_names):
        '''Glue layers to the model.
        
        Parameters
        ----------
            gluing_layers: List or tuple.
                Gluing layers.
            glued_layer_names: List or tuple.
                Glued layer names between beginning and ending.
        '''
        
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
        '''Create a progressive model for progressive learning.
        
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
        '''
        
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
            inputs =[Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in self.inputs]
            
            # Initial layers.
            x = inputs
            for idx in range(len(inputs), fixed_s_layer_depth + 1): #?
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
            inputs = [Input(K.int_shape(self.get_layer(layer_names[prog_depth]).input)[1:])] # Multiple inputs?
            
            # Initial and middle layers.
            x = inputs
            for idx in range(prog_depth, fixed_e_layer_depth + 1):
                x = self.get_layer(layer_names[idx])(x) # Input layer?
            
            # Final layers. ?
            if len(self.inputs) > 1:
                aug_inputs = [Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in self.inputs[1:]]
            
            for idx in range(fixed_e_layer_depth + 1, total_depth):
                layer = self.get_layer(layer_names[idx])
                if len(layer.inputs) > 1:
                    x = layer([x] + aug_inputs)
                else:
                    x = layer(x) 
                        
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
        '''Create a progressive model for progressive learning within the current model.
        
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
        '''
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
        '''Runs a single gradient update on a single batch of data for the forward progressive model.
        
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
        '''
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
        '''Runs a single gradient update on a single batch of data for the backward progressive model.
        
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
        '''
        assert hasattr(self, 'backward_prog_model') and self.backward_prog_model._is_compiled
        return self.backward_prog_model.train_on_batch(x
                                               , y=y 
                                               , sample_weight=sample_weight
                                               , class_weight=class_weight
                                               , reset_metrics=reset_metrics)
                                        