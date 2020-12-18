'''
Created on 2020. 4. 16.

@author: Inwoo Chung (gutomitai@gmail.com)
'''

import numpy as np
import pandas as pd
import os
from abc import ABC, abstractmethod
import time
import json
import platform

from tqdm import tqdm

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.utils import CustomObjectScope

#os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Constants.
DEBUG = True

def create_scaling_func(a, b):
    return lambda x: (b - a) * x + a

def policy_loss(y_true, y_pred):
    return y_pred

class Critic(ABC):
    """Abstract critic class."""
        
    @abstractmethod
    def __init__(self, hps, resource_path, model_loading, *args, **keywords):
        pass
    
    @abstractmethod
    def train(self, state, action, td_target):
        pass
    
    @abstractmethod
    def predict_action_value(self, state, action):
        pass

class Actor(ABC):
    """Abstract actor class."""
        
    @abstractmethod
    def __init__(self, hps, resource_path, model_loading, *args, **keywords):
        pass
    
    @abstractmethod
    def train(self, state, action, td_error):
        pass
    
    @abstractmethod
    def act(self, state):
        pass

class RLModel(ABC):
    """Abstract reinforcement learning model."""

    @abstractmethod
    def __init__(self, config_path, resource_path, model_loading, *args, **keywords):
        pass
    
    @abstractmethod
    def learn(self, *args, **keywords):
        pass
    
    @abstractmethod
    def act(self, *args, **keywords):
        pass

class Learner():
    """Abstract learner."""
    pass

class Trainer():
    """Abstract trainer."""
    pass

class StyleBasedGANTrainer(Trainer):
    """Style based GAN optimization via RL."""
    
    class OptCritic(Critic):
        """Critic."""
        
        # Constants.    
        MODEL_PATH = 'opt_critic.h5'
        
        def __init__(self, resource_path, conf):
            """
            Parameters
            ----------
            resource_path: String.
                Resource path.
            conf: Dictionary.
                Configuration.
            """
            
            # Initialize.
            self.resource_path = resource_path
            self.conf = conf
            self.hps = conf['hps']
            self.nn_arch = conf['nn_arch']
            self.model_loading = conf['model_loading']
                
            if self.model_loading:
                self.model = load_model(os.path.join(self.MODEL_PATH)) # Check exception.
            else:
                # Design action value function.            
                # Input.
                input_a = Input(shape=(self.nn_arch['action_dim'],), name='input_a')
                                
                # Get action value.
                x = input_a
                for i in range(self.nn_arch['num_layers']):
                    x = Dense(self.nn_arch['dense_layer_dim'], activation='relu', name='dense' + str(i + 1))(x)
                    
                action_value = Dense(1, activation='linear', name='action_value_dense')(x)

                self.model = Model(inputs=[input_a], outputs = [action_value], name='opt_critic')

                opt = optimizers.Adam(lr=self.hps['lr']
                                        , beta_1=self.hps['beta_1']
                                        , beta_2=self.hps['beta_2']
                                        , decay=self.hps['decay'])

                self.model.compile(optimizer=opt, loss='mse')
                #self.model.summary()
        
        def train(self, s, a, td_target): # learning rate?
            """Train critic.
            
            Parameters
            ----------
            a: 2D numpy array. 
                Action, a.
            td_target : 2D numpy array. 
                TD target array, batch size (value)?
            """
                                                    
            # Train model online.
            if self.conf['multi_gpu']:
                self.parallel_model.train_on_batch([a]
                           , [td_target])                
            else:
                self.model.train_on_batch([a]
                           , [td_target])
            
            # Save the model.
            self.model.save(os.path.join(self.MODEL_PATH))
    
        def predict_action_value(self, s, a):
            """Predict action value.
            
            Parameters
            ----------
            a: 2D numpy array. 
                Action, a.
            
            Returns
            -------
            Action value.
                2D numpy array.
            """
            
            # Predict action value.
            action_value = self.model.predict([a])
            
            return action_value
            
    class OptActor(Actor):
        """Actor."""
        
        # Constants.    
        MODEL_PATH = 'opt_actor.h5'
    
        def __init__(self, resource_path, conf):
            """
            Parameters
            ----------
            resource_path: String.
                Raw data path.
            conf: Dictionary.
                Configuration.
            """
            
            # Initialize.
            self.resource_path = resource_path
            self.conf = conf
            self.hps = conf['hps']
            self.nn_arch = conf['nn_arch']
            self.model_loading = conf['model_loading']
                
            if self.model_loading:
                with CustomObjectScope({'policy_loss': policy_loss}):
                    self.model = load_model(os.path.join(self.MODEL_PATH)) # Check exception.
            else:
                # Design actor.
                # Input.
                input_s = Input(shape=(self.nn_arch['state_dim'], ), name='input_s')
                
                # Get action.
                x = input_s
                for i in range(self.nn_arch['num_layers']):
                    x = Dense(self.nn_arch['dense_layer_dim'], activation='relu', name='dense' + str(i + 1))(x)
                    
                action = Dense(self.nn_arch['action_dim'], activation='tanh', name='action_value_dense')(x)
    
                input_td_error = Input(shape=(1,))
                action = Lambda(lambda x: K.log(x))(action) #?
                action = Lambda(lambda x: -1.0 * x[0] * x[1])([input_td_error, action])

                self.model = Model(inputs=[input_s, input_td_error], outputs = [action])
                opt = optimizers.Adam(lr=self.hps['lr']
                                        , beta_1=self.hps['beta_1']
                                        , beta_2=self.hps['beta_2']
                                        , decay=self.hps['decay'])

                self.model.compile(optimizer='adam', loss=policy_loss)
                #self.model.summary()
            
            self._make_action_model() 
       
        def _make_action_model(self):
            """Make action model."""
            input_s = Input(shape=(self.nn_arch['state_dim'], ), name='input_s')
            
            # Get action.
            x = input_s
            for i in range(self.nn_arch['num_layers']):
                x = self.model.get_layer('dense' + str(i + 1))(x)
                
            action = self.model.get_layer('action_value_dense')(x)
            self.action_model = Model(inputs=[input_s], outputs=[action])            
            
        def train(self, s, a, td_error):
            """Train actor.
            
            Parameters
            ----------
            s: 2D numpy array. 
                State, s.
            a: 2D numpy array. 
                Action, a.
            td_errors: 2D numpy array. 
                TD error value.
            """
            
            # Train.
            if self.conf['multi_gpu']:
                self.parallel_model.train_on_batch([s, td_error] # td_error dimension?
                           , [a])           
            else:
                self.model.train_on_batch([s, td_error] # td_error dimension?
                           , [a])            
    
            # Save the model.
            self.model.save(os.path.join(self.MODEL_PATH))
        
        def act(self, s): # Both same function?
            """Get hyper-parameters and neural network architecture information.
            
            Parameters
            ----------
            s: 2D numpy array. 
                State, s.
            
            Returns
            ------
            2D numpy array.
                Bound model configuration.
            """
            return self.action_model.predict(s) #?
    
    def __init__(self, config_path):
        """
        Parameters
        ----------
        config_path: String.
            Configuration file path.
        """
        
        # Initialize.
        np.random.seed(int(time.time()))
                
        # Load configuration.
        with open(os.path.join(config_path), 'r') as f:
            self.conf = json.load(f)

        self.resource_path = self.conf['resource_path']
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']
        self.critic_conf = self.conf['critic_conf']
        self.actor_conf = self.conf['actor_conf']
        
        # Instantiate critic, actor.
        self.critic = self.OptCritic(self.resource_path, self.critic_conf)
        self.actor= self.OptActor(self.resource_path, self.actor_conf)
                
        # Initial state and action.
        self.state = np.random.normal(size = (self.hps['batch_size'], self.nn_arch['state_dim'])) # Optimal initializer. ?
        self.action = self.actor.act(self.state)
            
    def learn(self, feedback):
        """Learn."""
                                                       
        # Train critic and actor for reward and state.                
        # Get rewards, states.
        state_p = feedback['state']
        reward = feedback['reward']
                    
        # Sample next actions.
        action_p = self.actor.act(state_p)
                    
        # Train. Dimension?
        td_target = reward + self.hps['gamma'] * self.critic.predict_action_value(state_p, action_p)
        td_error = td_target - self.critic.predict_action_value(self.state, self.action)

        self.critic.train(self.state, self.action, td_target) #?
        self.actor.train(self.state, self.action, td_error) #?
        
        self.state = state_p
        self.action = action_p
            
    def act(self, s): # Both same function?
        """Get a weight value.
        
        Parameters
        ----------
        s: 2D numpy array. 
            State, s.
        
        Returns
        ------
        Float32.
            A weight value.
        """
        return np.mean(self.actor.act(s), axis=0)

    def optimize(self, f_conf):
        """Optimize the style based GAN model via RL.""" 
        rs_mean = 1.       
        for i in tqdm(range(self.hps['steps'])):
            # Convert normalized hyper-parameters and NN architecture information to original values.
            action = (self.action + 1.0) * 0.5 # -1.0 ~ 1.0 -> 0.0 ~ 1.0.
            rs = []
            
            # Create scaling functions for each parameter.
            s_funcs = []
            s_funcs.append(create_scaling_func(2.0, 8.0)) # batch_size.
            s_funcs.append(create_scaling_func(100.0, 1000.0)) # lambda.
            s_funcs.append(create_scaling_func(1e-1, 1e-7)) # disc_ext_hps: lr.
            s_funcs.append(create_scaling_func(0.0, 1.0)) # disc_ext_hps: beta_1.
            s_funcs.append(create_scaling_func(0.0, 1.0)) # disc_ext_hps: beta_2.
            s_funcs.append(create_scaling_func(0.0, 1.0)) # disc_ext_hps: decay.
            s_funcs.append(create_scaling_func(1e-1, 1e-7)) # gen_disc_hps: lr.
            s_funcs.append(create_scaling_func(0.0, 1.0)) # gen_disc_hps: beta_1.
            s_funcs.append(create_scaling_func(0.0, 1.0)) # gen_disc_hps: beta_2.
            s_funcs.append(create_scaling_func(0.0, 1.0)) # gen_disc_hps: decay.
            
            for j in range(self.hps['batch_size']):
                # hps.
                f_conf['hps']['batch_size'] = int(s_funcs[0](action[j][0])) * inpainting.NUM_PUNCHED_IMAGES_PER_IMAGE
                f_conf['hps']['lambda'] = s_funcs[1](action[j][1])
                
                # disc_ext_hps.
                f_conf['disc_ext_hps']['lr'] = s_funcs[2](action[j][2])
                f_conf['disc_ext_hps']['beta_1'] = s_funcs[3](action[j][3])
                f_conf['disc_ext_hps']['beta_2'] = s_funcs[4](action[j][4])
                f_conf['disc_ext_hps']['decay'] = s_funcs[5](action[j][5])
                
                # disc_ext_hps.
                f_conf['gen_disc_hps']['lr'] = s_funcs[6](action[j][6])
                f_conf['gen_disc_hps']['beta_1'] = s_funcs[7](action[j][7])
                f_conf['gen_disc_hps']['beta_2'] = s_funcs[8](action[j][8])
                f_conf['gen_disc_hps']['decay'] = s_funcs[9](action[j][9])   
    
                # Train.
                cf = COVID19Forecastor(f_conf)
            
                ts = time.time()
                cf.train()
                te = time.time()
            
                #print('Elapsed time: {0:f}s'.format(te-ts))
            
                # Calculate reward.
                r = -1.0 * cf.evaluate()
                
                # Check exception.
                if np.isnan(r):
                    continue
                
                rs.append(r)
                
            rs = np.asarray(rs)
            
            # Check exception.
            if len(rs) == 0:
                self.state = np.random.normal(size =(self.hps['batch_size'], self.nn_arch['state_dim'])) # Optimal initializer. ?
                self.action = self.actor.act(self.state)
                continue
            
            if rs < rs_mean:
                print('Save the model.')            
                cf.model.save(os.path.join(self.resource_path, cf.MODEL_FILE_NAME))
                rs = rs.mean()
            
            print(f_conf, rs.mean())
                
            self.state = np.random.normal(size =(self.hps['batch_size'], self.nn_arch['state_dim'])) # Optimal initializer. ?
            feedback = {'state': self.state, 'reward': rs}            
            self.learn(feedback)
            self.action = self.actor.act(self.state)                    
        
def main():
    # Optimize the style based GAN model via RL.
    # Create the optimization RL entity.
    config_path = 'style_based_gan_opt_via_rl_conf.json'
    style_GAN_opt = StyleBasedGANTrainer(config_path)
    
    with open("style_based_gan_conf.json", 'r') as f:
        f_conf = json.load(f)  
    
    style_GAN_opt.optimize(f_conf)
    
if __name__ == '__main__':
    main()