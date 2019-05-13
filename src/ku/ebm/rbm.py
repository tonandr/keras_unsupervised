"""
Created on 2019. 5. 11.

@author: Inwoo Chung (gutomitai@gmail.com)
License: BSD 3 clause.

Revision:
    -May 13, 2019
        RBM class's main functions has been developed.
"""

from keras import backend as K
from keras.layers import Layer, Input
from tensorflow.keras import initializers

class RBM(Layer):
    """Restricted Boltzmann Machine based on Keras."""
    def __init__(self, hps, output_dim, **kwargs):
        self.hps = hps
        self.output_dim = output_dim
        super(RBM, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.rbm_weight = self.add_weight(name='rbm_weight'
                                 , shape=(input_shape[1], self.output_dim)
                                 , initializer='uniform' # Which initializer is optimal?
                                 , trainable=True)
        super(RBM, self).build(input_shape)

        self.hidden_bias = K.variable(initializers.get('uniform')((self.output_dim, ))
                            , dtype=K.floatx()
                            , name='rbm_hidden_bias')
        self.visible_bias = K.variable(initializers.get('uniform')((self.output_dim, ))
                            , dtype=K.floatx()
                            , name='rbm_visible_bias')
        
        # Make symbolic computation objects.
        # Transform visible units.
        input_visible = K.placeholder(shape=(None, input_shape[1]), name='input_visible')
        transform = K.sigmoid(K.dot(input_visible, self.rbm_weight) + self.hidden_bias)
        self.transform_func = K.function([input_visible], [transform])
  
        # Transform hidden units.      
        input_hidden = K.placeholder(shape=(None, self.output_dim), name='input_hidden')
        inv_transform = K.sigmoid(K.dot(input_hidden, K.transpose(self.rbm_weight)) + self.visible_bias)
        self.inv_transform_func = K.function([input_hidden], [inv_transform])
        
        # Contrastive divergence.
        v_pos = input_visible
        h_pos = transform
        v_neg = K.less(K.random_uniform(shape=(None, input_shape[1]))
                       , K.sigmoid(K.dot(h_pos, K.transpose(self.rbm_weight)) + self.visible_bias))
        h_neg = K.sigmoid(K.dot(v_neg, self.rbm_weight) + self.visible_bias)
        update = K.transpose(K.dot(K.transpose(v_pos), h_pos)) - K.dot(K.transpose(h_neg), v_neg)
        self.rbm_weight_update_func = K.function([input_visible], 
                                [K.update_add(self.rbm_weight, self.hps['lr'] * update)])
        self.hidden_bias_update_func = K.function([input_visible], 
                                [K.update_add(self.hidden_bias, self.hps['lr'] * (K.sum(h_pos) - K.sum(h_neg)))])
        self.visible_bias_update_func = K.function([input_visible], 
                                [K.update_add(self.visible_bias, self.hps['lr'] * (K.sum(v_pos) - K.sum(v_neg)))])        

    def call(self, x):
        return K.sigmoid(K.dot(x, self.rbm_weight) + self.hidden_bias)
    
    def transform(self, v):
        return self.transform_func(v)
    
    def inv_transform(self, h):
        return self.inv_transform_func(h)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def fit(self, V, batch_size=1):
        """Train RBM with the data V.
        
        Parameters
        ----------
        V : 2d numpy array
            Visible data (batch size x input_dim).
        batch_size : integer
            Batch size (default, 1).
        """
        num_step = V.shape[0] // batch_size \
            if V.shape[0] % batch_size == 0 else V.shape[0] // batch_size + 1 # Exception processing?
        
        for i in range(num_step):
            if i == (num_step - 1):
                V_batch = V[int(i*batch_size):V.shape[0]]
                
                # Train.
                self.rbm_weight_update_func(V_batch)
                self.hidden_bias_update_func(V_batch)
                self.visible_bias_update_func(V_batch)
            else:
                V_batch = V[int(i*batch_size):int((i+1)*batch_size)]
                
                # Train.
                self.rbm_weight_update_func(V_batch)
                self.hidden_bias_update_func(V_batch)
                self.visible_bias_update_func(V_batch)       