from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import initializers

from ku.backend_ext import tensorflow_backend as Ke

# Constants.
MODE_VISIBLE_BERNOULLI = 0
MODE_VISIBLE_GAUSSIAN = 1
MODE_COMPLEX = 2 # TODO

class DBN(object):
    """Deep belief network."""
    
    def add_stack(self, rbm_layer):
        """Add a rbm layer to the dbn stack.
        
        Parameters
        ----------
        rbm_layer: RBM class
            RBM layer.
        """
        
        # Check exception?
        if hasattr(self, '_rbm_layers') \
            and self._rbm_layers[-1].output_shape[1] == self.rbm_layer.input_shape[1]:
            self._rbm_layers.append(rbm_layer)
        elif hasattr(self, 'rbm_layers') \
            and self._rbm_layers[-1].output_shape[1] != self.rbm_layer.input_shape[1]:
            raise ValueError('A previous RBM layer\'s output dimension must' \
                             + 'be equal to a next one\'s input dimension.')
        else:
            self._rbm_layers = [rbm_layer]
    
    def fit(self, V, verbose=1):
        """Train DBN with the data V.
        
        Parameters
        ----------
        V: 2d numpy array
            Visible data (batch size x input_dim).
        verbose: integer
            Verbose mode (default, 1).
        """
        V_p = V.copy()
        
        # Check exception?
        if hasattr(self, '_rbm_layers') != True:
            raise ValueError('Any rbm layer doesn\'t exist.')
        
        # Train each rbm layer.
        for rbm_layer in self._rbm_layers:
            # RBM training.
            print('Train {0:s}.'.format(rbm_layer.name))
            self.rbm_layer.fit(V_p)
            V_p = self.rbm_layer.transform(V_p)
            
    def transform(self, V):
        """Transform the visible unit.

        Parameters
        ----------
        V: 2d numpy array
            Visible data (batch size x input_dim).
        """
        V_p = V.copy()
        
        # Check exception?
        if hasattr(self, '_rbm_layers') != True:
            raise ValueError('Any rbm layer doesn\'t exist.')
        
        # Train each rbm layer.
        for rbm_layer in self._rbm_layers:
            V_p = rbm_layer.transform(V_p)                
        
        return V_p
            
    def inv_transform(self, H):
        """Transform the hidden unit.

        Parameters
        ----------
        H: 2d numpy array
            Hidden data (batch size x output_dim).
        """
        H_p = H.copy()
        
        # Check exception?
        if hasattr(self, '_rbm_layers') != True:
            raise ValueError('Any rbm layer doesn\'t exist.')
        
        # Train each rbm layer.
        for i in range(len(self._rbm_layers), -1):
            rbm_layer = self._rbm_layers[i]
            H_p = self.rbm_layer.inv_transform(H_p)                
        
        return H_p                  