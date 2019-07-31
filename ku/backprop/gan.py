"""
Created on 2019. 6. 17.

@author: Inwoo Chung (gutomitai@gmail.com)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC, abstractmethod
from functools import partial

from keras.layers import Input, Lambda
from keras.models import Model, load_model
from keras.utils import multi_gpu_model
from keras import optimizers
from keras.utils.generic_utils import CustomObjectScope

from ..layer_ext import InputRandomUniform
from ..loss_ext import gen_disc_wgan_loss, disc_ext_wgan_loss, disc_ext_wgan_gp_loss

# GAN mode.
REGULAR_GAN = 0
W_GAN_GP = 1

class AbstractGAN(ABC):
    """Abstract generative adversarial network."""

    # Constants.
    GEN_DISC_PATH = 'gen_disc.h5'
    DISC_EXT_PATH = 'disc_ext.h5'  
    
    def __init__(self, conf):
        """
        Parameters
        ----------
        conf: dict
            Configuration.
        """
        self.conf = conf #?
        
        if self.conf['gan_mode'] == REGULAR_GAN:
            pass
        elif self.conf['gan_mode'] == W_GAN_GP:
            if self.conf['model_loading']:
                if not hasattr(self, 'custom_objects'):
                    RuntimeError('Before models, custom_objects must be created.')
                
                self.custom_objects['gen_disc_wgan_loss'] = gen_disc_wgan_loss
                                                            
                with CustomObjectScope(self.custom_objects): 
                    # gen_disc.
                    self.gen_disc = load_model(self.GEN_DISC_PATH, compile=False)
                                            
                    # gen.
                    self.gen = self.gen_disc.get_layer('gen')
                    
                    # disc.
                    self.disc = self.gen_disc.get_layer('disc')

                self.compile() #?                   
        else:
            raise ValueError('The valid gan mode must be assigned.')
                                    
    @abstractmethod
    def _create_generator(self):
        """Create the generator."""
        pass
    
    @abstractmethod        
    def _create_discriminator(self):
        """Create the discriminator."""
        pass
    
    def compile(self):
        """Create the GAN model and compile it."""
        
        # Check exception?
        if hasattr(self, 'gen') != True or hasattr(self, 'disc') != True:
            raise RuntimeError('The generator and discriminator must be created')
        
        if self.conf['gan_mode'] == REGULAR_GAN:
            #self._compile_regular_gan()
            pass
        elif self.conf['gan_mode'] == W_GAN_GP:
            self._compile_wgan_gp()
        else:
            raise ValueError('The valid gan mode must be assigned.')

    '''
    def _compile_regular_gan(self):
        """Compile regular gan."""
        
        # Design gan according to input and output nodes for each model.        
        # Design and compile disc_ext.
        x_inputs = self.disc.inputs if self.nn_arch['label_usage'] else [self.disc.inputs]  
        x_outputs = [self.disc(x_inputs)]
        
        z_inputs = self.gen.inputs
        
        if self.conf['multi_gpu']:
            self.gen_p = multi_gpu_model(self.gen, gpus=self.conf['num_gpus'])
               
        self.gen.trainable = False #?
        self.gen.name ='gen'    
        z_outputs = self.gen(z_inputs) if self.nn_arch['label_usage'] else [self.gen(z_inputs)]
        self.disc.name = 'disc'
        x2_outputs = [self.disc(z_outputs)]
        
        self.disc_ext = Model(inputs=x_inputs + z_inputs, outputs=x_outputs + x2_outputs)        
    
        opt = optimizers.Adam(lr=self.hps['lr']
                                    , beta_1=self.hps['beta_1']
                                    , beta_2=self.hps['beta_2']
                                    , decay=self.hps['decay'])

        if hasattr(self, 'disc_ext_losses') == False \
            or hasattr(self, 'disc_ext_loss_weights') == False:
            raise RuntimeError("disc_ext_losses and disc_ext_weights must be created.") #?
                    
        self.disc_ext.compile(optimizer=opt
                         , loss=self.disc_ext_losses
                         , loss_weights=self.disc_ext_loss_weights)
        
        if self.conf['multi_gpu']:
            self.disc_ext_p = multi_gpu_model(self.disc_ext, gpus=self.conf['num_gpus'])
            self.disc_ext_p.compile(optimizer=opt
                                    , loss=self.disc_ext.losses
                                    , loss_weights=self.disc_ext_loss_weights) #?                     
               
        # Design and compile gen_disc.
        z_inputs = self.gen.inputs
        self.gen.trainable = True #?     
        z_outputs = self.gen(z_inputs) if self.nn_arch['label_usage'] else [self.gen(z_inputs)]
        self.disc.trainable = False #?
        z_p_outputs = [self.disc(z_outputs)] #?

        self.gen_disc = Model(inputs=z_inputs, outputs=z_p_outputs)
        
        if hasattr(self, 'gen_disc_losses') == False \
            or hasattr(self, 'gen_disc_loss_weights') == False:
            raise RuntimeError("gen_disc_losses and gen_disc_weights must be created.") #?
        
        self.gen_disc.compile(optimizer=opt
                         , loss=self.gen_disc_losses #?
                         , loss_weights=self.gen_disc_loss_weights)

        if self.conf['multi_gpu']:
            self.gen_disc_p = multi_gpu_model(self.gen_disc, gpus=self.conf['num_gpus'])
            self.gen_disc_p.compile(optimizer=opt
                         , loss=self.gen_disc_losses #?
                         , loss_weights=self.gen_disc_loss_weights)
    '''

    def _compile_wgan_gp(self):
        """Compile wgan_gp."""
        
        # Design gan according to input and output nodes for each model.        
        # Design and compile disc_ext.
        # x.
        x_inputs = self.disc.inputs if self.nn_arch['label_usage'] else [self.disc.inputs]  
        x_outputs = [self.disc(x_inputs)]
        
        # x_tilda.
        z_inputs = self.gen.inputs
        
        if self.conf['multi_gpu']:
            self.gen_p = multi_gpu_model(self.gen, gpus=self.conf['num_gpus'])
               
        self.gen.trainable = False #?
        self.gen.name ='gen'    
        z_outputs = self.gen(z_inputs) if self.nn_arch['label_usage'] else [self.gen(z_inputs)]
        self.disc.name = 'disc'
        x2_outputs = [self.disc(z_outputs)]
        
        # x_hat.
        x3_inputs = [Input(shape=(1, ))] # Trivial input.
        x3_epsilon = [InputRandomUniform((1, 1, 1))(x3_inputs)]
        x3 = Lambda(lambda x: x[2] * x[1] + (1.0 - x[2]) * x[0])([x_inputs[0]] + [z_outputs[0]] + x3_epsilon) #?
        #x3 = Lambda(lambda x: x[0])([x_inputs[0]] + [z_outputs[0]] + x3_epsilon) #?
        x3_outputs = [self.disc([x3, x_inputs[1]])]
        
        self.disc_ext = Model(inputs=x_inputs + z_inputs + x3_inputs
                              , outputs=x_outputs + x2_outputs + x3_outputs)        
    
        opt = optimizers.Adam(lr=self.hps['lr']
                                    , beta_1=self.hps['beta_1']
                                    , beta_2=self.hps['beta_2']
                                    , decay=self.hps['decay'])

        # Make losses.        
        self.disc_ext_losses = [disc_ext_wgan_loss
                                , disc_ext_wgan_loss
                                , disc_ext_wgan_gp_loss(input_variables=x3
                                                        , wgan_lambda=self.conf['hps']['wgan_lambda']
                                                        , wgan_target=self.conf['hps']['wgan_target'])]
        self.disc_ext_loss_weights = [-1.0, 1.0, 1.0]

        if hasattr(self, 'disc_ext_losses') == False \
            or hasattr(self, 'disc_ext_loss_weights') == False:
            raise RuntimeError("disc_ext_losses and disc_ext_weights must be created.") #?
                    
        self.disc_ext.compile(optimizer=opt
                         , loss=self.disc_ext_losses
                         , loss_weights=self.disc_ext_loss_weights)
        
        if self.conf['multi_gpu']:
            self.disc_ext_p = multi_gpu_model(self.disc_ext, gpus=self.conf['num_gpus'])
            self.disc_ext_p.compile(optimizer=opt
                                    , loss=self.disc_ext.losses
                                    , loss_weights=self.disc_ext_loss_weights) #?                     
               
        # Design and compile gen_disc.
        z_inputs = self.gen.inputs
        self.gen.trainable = True #?     
        z_outputs = self.gen(z_inputs) if self.nn_arch['label_usage'] else [self.gen(z_inputs)]
        self.disc.trainable = False #?
        z_p_outputs = [self.disc(z_outputs)] #?

        self.gen_disc = Model(inputs=z_inputs, outputs=z_p_outputs)
        
        # Make losses.
        self.gen_disc_losses = [gen_disc_wgan_loss]
        self.gen_disc_loss_weights = [-1.0]
        
        if hasattr(self, 'gen_disc_losses') == False \
            or hasattr(self, 'gen_disc_loss_weights') == False:
            raise RuntimeError("gen_disc_losses and gen_disc_weights must be created.") #?
                
        self.gen_disc.compile(optimizer=opt
                         , loss=self.gen_disc_losses #?
                         , loss_weights=self.gen_disc_loss_weights)

        if self.conf['multi_gpu']:
            self.gen_disc_p = multi_gpu_model(self.gen_disc, gpus=self.conf['num_gpus'])
            self.gen_disc_p.compile(optimizer=opt
                         , loss=self.gen_disc_losses #?
                         , loss_weights=self.gen_disc_loss_weights)        
    
    @abstractmethod                        
    def fit(self, x_inputs, x_outputs):
        """Train the GAN model.
        
        Parameters
        ----------
        x_inputs : list.
            Training data numpy array list.
        x_outputs : list.
            Ground truth data numpy array list.
        """
        pass

    @abstractmethod  
    def fit_generator(self
                      , generator
                      , max_queue_size=10
                      , workers=1
                      , use_multiprocessing=False
                      , shuffle=True):
        """Train the GAN model with the generator.
        
        Parameters
        ----------
        generator: Generator
            Training data generator.
        max_queue_size: Integer
            Maximum size for the generator queue (default: 10).
        workers: Integer
            Maximum number of processes to get samples (default: 1, 0: main thread).
        use_multiprocessing: Boolean
            Multi-processing flag (default: False).
        shuffle: Boolean
            Batch shuffling flag (default: True).
        """
        pass
 
    @abstractmethod    
    def generate(self, *args, **kwargs):
        """Generate styled images."""
        pass 