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

class RBM(Layer):
    """Restricted Boltzmann machine based on Keras."""
    
    def __init__(self, hps, output_dim, name=None, mode=MODE_VISIBLE_GAUSSIAN, **kwargs):
        self.hps = hps
        self.output_dim = output_dim
        self.name = name
        self.mode = mode
        super(RBM, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.rbm_weight = self.add_weight(name='rbm_weight'
                                 , shape=(input_shape[1], self.output_dim)
                                 , initializer='uniform' # Which initializer is optimal?
                                 , trainable=True)
        self.hidden_bias = self.add_weight(name='rbm_hidden_bias'
                                           , shape=(self.output_dim, )
                                           , initializer='uniform'
                                           , trainable=True)
        self.visible_bias = K.variable(initializers.get('uniform')((input_shape[1], ))
                            , dtype=K.floatx()
                            , name='rbm_visible_bias')
        
        # Make symbolic computation objects.
        if self.mode == MODE_VISIBLE_BERNOULLI: 
            # Transform visible units.
            self.input_visible = K.placeholder(shape=(None, input_shape[1]), name='input_visible')
            self.transform = K.cast(K.less(K.random_uniform(shape=(self.hps['batch_size'], input_shape[1]))
                , K.sigmoid(K.dot(self.input_visible, self.rbm_weight) + self.hidden_bias)))
            self.transform_func = K.function([self.input_visible], [self.transform])
      
            # Transform hidden units.      
            self.input_hidden = K.placeholder(shape=(None, self.output_dim), name='input_hidden')
            self.inv_transform = K.cast(K.less(K.random_uniform(shape=(self.hps['batch_size'], input_shape[1]))
                , K.sigmoid(K.dot(self.input_hidden, K.transpose(self.rbm_weight)) + self.visible_bias)))
            self.inv_transform_func = K.function([self.input_hidden], [self.inv_transform])
        elif self.mode == MODE_VISIBLE_GAUSSIAN:
            # Transform visible units.
            self.input_visible = K.placeholder(shape=(None, input_shape[1]), name='input_visible')
            self.transform =  K.cast(K.less(K.random_uniform(shape=(self.hps['batch_size'], input_shape[1]))
                , K.relu(K.dot(self.input_visible, self.rbm_weight) + self.hidden_bias))) #?
            self.transform_func = K.function([self.input_visible], [self.transform])
      
            # Transform hidden units.      
            self.input_hidden = K.placeholder(shape=(None, self.output_dim), name='input_hidden')
            self.inv_transform = Ke.multivariate_normal_diag(loc=(K.dot(self.input_hidden
                , K.transpose(self.rbm_weight)) + self.visible_bias)
                , scale_diag=np.ones(shape=(self.hps['batch_size'], input_shape[1]))).sample()
            self.inv_transform_func = K.function([self.input_hidden], [self.inv_transform])            
        else:
            # TODO
            pass            
        
        # Calculate free energy. #?
        self.free_energy = -1 * (K.squeeze(K.dot(self.input_visible, K.expand_dims(self.visible_bias, axis=-1)), -1) +\
                                K.sum(K.log(1 + K.exp(K.dot(self.input_visible, self.rbm_weight) +\
                                                self.hidden_bias)), axis=-1))
        self.free_energy_func = K.function([self.input_visible], [self.free_energy])

        super(RBM, self).build(input_shape)
        
    def call(self, x):
        if self.mode == MODE_VISIBLE_BERNOULLI: 
            return K.cast(K.less(K.random_uniform(shape=(self.hps['batch_size'], x.shape[1])) #?
                , K.sigmoid(K.dot(x, self.rbm_weight) + self.hidden_bias)))
        elif self.mode == MODE_VISIBLE_GAUSSIAN:
            return K.cast(K.less(K.random_uniform(shape=(self.hps['batch_size'], x.shape[1]))
                , K.relu(K.dot(x, self.rbm_weight) + self.hidden_bias))) #?
    
    def transform(self, v):
        return self.transform_func(v)
    
    def inv_transform(self, h):
        return self.inv_transform_func(h)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def cal_free_energy(self, v):
        return self.free_energy_func(v) #?
    
    def fit(self, V, verbose=1):
        """Train RBM with the data V.
        
        Parameters
        ----------
        V : 2d numpy array
            Visible data (batch size x input_dim).
        verbose : integer
            Verbose mode (default, 1).
        """
        num_step = V.shape[0] // self.hps['batch_size'] \
            if V.shape[0] % self.hps['batch_size'] == 0 else V.shape[0] // self.hps['batch_size'] + 1 # Exception processing?
             
        for k in range(self.hps['epochs']):
            if verbose == 1:
                print(k + 1, '/', self.hps['epochs'], ' epochs', end='\r')

            if self.mode == MODE_VISIBLE_BERNOULLI: 
                # Contrastive divergence.
                v_pos = self.input_visible
                h_pos = self.transform
                v_neg = K.cast(K.less(K.random_uniform(shape=(self.hps['batch_size'], V.shape[1]))
                        , K.sigmoid(K.dot(h_pos, K.transpose(self.rbm_weight)) + self.visible_bias))
                        , dtype=np.float32)
                h_neg = K.sigmoid(K.dot(v_neg, self.rbm_weight) + self.hidden_bias)
                update = K.transpose(K.transpose(K.dot(K.transpose(v_pos), h_pos)) \
                                     - K.dot(K.transpose(h_neg), v_neg))
                self.rbm_weight_update_func = K.function([self.input_visible]
                                                , [K.update_add(self.rbm_weight, self.hps['lr'] * update)])
                self.hidden_bias_update_func = K.function([self.input_visible]
                                                , [K.update_add(self.hidden_bias, self.hps['lr'] \
                                                * (K.sum(h_pos, axis=0) - K.sum(h_neg, axis=0)))])
                self.visible_bias_update_func = K.function([self.input_visible]
                                                , [K.update_add(self.visible_bias, self.hps['lr'] \
                                                * (K.sum(v_pos, axis=0) - K.sum(v_neg, axis=0)))])
                
                # Create the fist visible nodes sampling object.
                self.sample_first_visible = K.function([self.input_visible]
                                                    , [v_neg])
            elif self.mode == MODE_VISIBLE_GAUSSIAN:
                # Contrastive divergence.
                v_pos = self.input_visible
                h_pos = self.transform
                v_neg = Ke.multivariate_normal_diag(loc=(K.dot(h_pos, K.transpose(self.rbm_weight)) + self.visible_bias)
                                , scale_diag=np.ones(shape=(self.hps['batch_size'], V.shape[1]))).sample()
                h_neg = K.sigmoid(K.dot(v_neg, self.rbm_weight) + self.hidden_bias)
                update = K.transpose(K.transpose(K.dot(K.transpose(v_pos), h_pos)) \
                                     - K.dot(K.transpose(h_neg), v_neg))
                self.rbm_weight_update_func = K.function([self.input_visible]
                                                , [K.update_add(self.rbm_weight, self.hps['lr'] * update)])
                self.hidden_bias_update_func = K.function([self.input_visible]
                                                , [K.update_add(self.hidden_bias, self.hps['lr'] \
                                                * (K.sum(h_pos, axis=0) - K.sum(h_neg, axis=0)))])
                self.visible_bias_update_func = K.function([self.input_visible]
                                                , [K.update_add(self.visible_bias, self.hps['lr'] \
                                                * (K.sum(v_pos, axis=0) - K.sum(v_neg, axis=0)))])
                
                # Create the fist visible nodes sampling object.
                self.sample_first_visible = K.function([self.input_visible]
                                                    , [v_neg])
            else:
                pass                 
                   
            for i in range(num_step):
                if i == (num_step - 1):
                    if self.mode == MODE_VISIBLE_BERNOULLI: 
                        # Contrastive divergence.
                        v_pos = self.input_visible
                        h_pos = self.transform
                        v_neg = K.cast(K.less(K.random_uniform(shape=(V.shape[0] - int(i*self.hps['batch_size'], V.shape[1])))
                                , K.sigmoid(K.dot(h_pos, K.transpose(self.rbm_weight)) + self.visible_bias))
                                , dtype=np.float32)
                        h_neg = K.sigmoid(K.dot(v_neg, self.rbm_weight) + self.hidden_bias)
                        update = K.transpose(K.transpose(K.dot(K.transpose(v_pos), h_pos)) \
                                             - K.dot(K.transpose(h_neg), v_neg))
                        self.rbm_weight_update_func = K.function([self.input_visible]
                                                        , [K.update_add(self.rbm_weight, self.hps['lr'] * update)])
                        self.hidden_bias_update_func = K.function([self.input_visible]
                                                        , [K.update_add(self.hidden_bias, self.hps['lr'] \
                                                        * (K.sum(h_pos, axis=0) - K.sum(h_neg, axis=0)))])
                        self.visible_bias_update_func = K.function([self.input_visible]
                                                        , [K.update_add(self.visible_bias, self.hps['lr'] \
                                                        * (K.sum(v_pos, axis=0) - K.sum(v_neg, axis=0)))])
                        
                        # Create the fist visible nodes sampling object.
                        self.sample_first_visible = K.function([self.input_visible]
                                                            , [v_neg])
                    elif self.mode == MODE_VISIBLE_GAUSSIAN:
                        # Contrastive divergence.
                        v_pos = self.input_visible
                        h_pos = self.transform
                        v_neg = Ke.multivariate_normal_diag(loc=(K.dot(h_pos, K.transpose(self.rbm_weight)) + self.visible_bias)
                                    , scale_diag=np.ones(shape=(V.shape[0] - int(i*self.hps['batch_size'], V.shape[1])))).sample()
                        h_neg = K.sigmoid(K.dot(v_neg, self.rbm_weight) + self.hidden_bias)
                        update = K.transpose(K.transpose(K.dot(K.transpose(v_pos), h_pos)) \
                                             - K.dot(K.transpose(h_neg), v_neg))
                        self.rbm_weight_update_func = K.function([self.input_visible]
                                                        , [K.update_add(self.rbm_weight, self.hps['lr'] * update)])
                        self.hidden_bias_update_func = K.function([self.input_visible]
                                                        , [K.update_add(self.hidden_bias, self.hps['lr'] \
                                                        * (K.sum(h_pos, axis=0) - K.sum(h_neg, axis=0)))])
                        self.visible_bias_update_func = K.function([self.input_visible]
                                                        , [K.update_add(self.visible_bias, self.hps['lr'] \
                                                        * (K.sum(v_pos, axis=0) - K.sum(v_neg, axis=0)))])
                        
                        # Create the fist visible nodes sampling object.
                        self.sample_first_visible = K.function([self.input_visible]
                                                            , [v_neg])
                    else:
                        pass  
                    
                    V_batch = [V[int(i*self.hps['batch_size']):V.shape[0]]]
                    
                    # Train.
                    self.rbm_weight_update_func(V_batch)
                    self.hidden_bias_update_func(V_batch)
                    self.visible_bias_update_func(V_batch)
                else:
                    V_batch = [V[int(i*self.hps['batch_size']):int((i+1)*self.hps['batch_size'])]]
                    
                    # Train.
                    self.rbm_weight_update_func(V_batch)
                    self.hidden_bias_update_func(V_batch)
                    self.visible_bias_update_func(V_batch)
            
                # Calculate a training score by each step.
                # Free energy of the input visible nodes.
                fe = self.cal_free_energy(V_batch)
                
                # Free energy of the first sampled visible nodes.
                V_p_batch = self.sample_first_visible(V_batch)
                fe_p = self.cal_free_energy(V_p_batch)
                
                score = np.mean(np.abs(fe[0] - fe_p[0])) # Scale?
                print('\n{0:d}/{1:d}, score: {2:f}'.format(i + 1, num_step, score))
    
    def get_config(self):
        """Get configuration."""
        config = {'hps': self.hps
                  , 'output_dim': self.output_dim
                  , 'name': self.name}
        base_config = super(RBM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
                