from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras.backend as K 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, Conv3D\
    , SeparableConv1D, SeparableConv2D, Conv2DTranspose, Conv3DTranspose\
    , Activation, BatchNormalization, InputLayer, Flatten, LeakyReLU\
    , Concatenate, Lambda, UpSampling1D, UpSampling2D

from ..composite_layer import DenseBatchNormalization
from ..gnn_layer import GraphConvolutionNetwork


def reverse_model(model):
    """Reverse a model.
    
    Shared layer, multiple nodes ?
    
    Parameters
    ----------
    model: Keras model
        Model instance.
    
    Returns
    -------
    Keras model
        Reversed model instance.
    """
    
    # Check exception.
    layers = model.layers
    output_layer = layers[-1]

    if isinstance(output_layer.output, list):
        raise RuntimeError('Output must not be list.')
    
    # Get all layers and extract the input layer and output layer.
    input_r = tf.keras.Input(shape=K.int_shape(output_layer.output)[1:])  
    
    # Reconstruct the model reversely.
    output = _get_reversed_outputs(output_layer, input_r)
    
    return Model(inputs=input_r, outputs=output)


def _get_reversed_outputs(output_layer, input_r):
    """Get reverse outputs recursively. ?
    
    Parameters
    ----------
    output_layer: Keras layer.
        Last layer of a model.
    input_r: Tensor.
        Reversed input.
    """
    
    # Check exception.?
    # TODO
    
    in_node = output_layer.inbound_nodes[0]
    out_layer = in_node.outbound_layer
    
    if isinstance(out_layer, InputLayer):
        output = input_r
        return output    
    elif isinstance(out_layer, Dense):
        output = Dense(out_layer.input_shape[1]
                       , activation=out_layer.activation
                       , use_bias=out_layer.use_bias)(input_r) #?
        
        # Get an upper layer.
        upper_layer = in_node.inbound_layers
        return _get_reversed_outputs(upper_layer, output)
    elif isinstance(out_layer, DenseBatchNormalization):
        dense = Dense(out_layer.dense_1.input_shape[1]
                  , activation=out_layer.dense_1.activation
                  , use_bias=out_layer.dense_1.use_bias)
        batchnormalization = BatchNormalization()
        if out_layer.activation_1 is not None:
            activation = out_layer.activation_1
        else:
            activation = None
        if out_layer.dropout_1 is not None:
            dropout = out_layer.dropout_1
        else:
            dropout = None
        dense_batchnormalization = DenseBatchNormalization(dense
                                                           , batchnormalization
                                                           , activation=activation
                                                           , dropout=dropout)
        output = dense_batchnormalization(input_r)

        # Get an upper layer.
        upper_layer = in_node.inbound_layers
        return _get_reversed_outputs(upper_layer, output)
    elif isinstance(out_layer, (Conv1D, SeparableConv1D)): #?
        if out_layer.strides[0] >= 2:
            output = UpSampling1D(size=out_layer.strides[0])(input_r)
        else:
            if isinstance(out_layer, Conv1D):
                output = Conv1D(out_layer.input_shape[-1]
                                , out_layer.kernel_size
                                , strides=1
                                , padding='same'  # ?
                                , activation=out_layer.activation
                                , use_bias=out_layer.use_bias)(input_r)  # ?
            elif isinstance(out_layer, SeparableConv1D):
                output = SeparableConv1D(out_layer.input_shape[-1]
                                , out_layer.kernel_size
                                , strides=1
                                , padding='same'  # ?
                                , activation=out_layer.activation
                                , use_bias=out_layer.use_bias)(input_r)  # ?

        # Get an upper layer.
        upper_layer = in_node.inbound_layers
        return _get_reversed_outputs(upper_layer, output)
    elif isinstance(out_layer, (Conv2D, SeparableConv2D)):        
        if out_layer.strides[0] >= 2 or out_layer.strides[1] >= 2:
            output = Conv2DTranspose(out_layer.input_shape[-1]
                                     , out_layer.kernel_size
                                     , strides=out_layer.strides
                                     , padding='same' #?
                                     , activation=out_layer.activation
                                     , use_bias=out_layer.use_bias)(input_r) #?
            #output = UpSampling2D()(input_r) #?
        else:
            if isinstance(out_layer, Conv2D):
                output = Conv2D(out_layer.input_shape[-1]
                                , out_layer.kernel_size
                                , strides=1
                                , padding='same'  # ?
                                , activation=out_layer.activation
                                , use_bias=out_layer.use_bias)(input_r)  # ?
            elif isinstance(out_layer, SeparableConv2D):
                output = SeparableConv2D(out_layer.input_shape[-1]
                                , out_layer.kernel_size
                                , strides=1
                                , padding='same'  # ?
                                , activation=out_layer.activation
                                , use_bias=out_layer.use_bias)(input_r)  # ?
        
        # Get an upper layer.
        upper_layer = in_node.inbound_layers
        return _get_reversed_outputs(upper_layer, output)
    elif isinstance(out_layer, (Conv3D)):
        output = Conv3DTranspose(out_layer.input_shape[-1]
                                     , out_layer.kernel_size
                                     , strides=out_layer.strides
                                     , padding='same' #?
                                     , activation=out_layer.activation
                                     , use_bias=out_layer.use_bias)(input_r) #?
        # output = UpSampling3D()(input_r) #?
            
        # Get an upper layer.
        upper_layer = in_node.inbound_layers
        return _get_reversed_outputs(upper_layer, output)
    elif isinstance(out_layer, GraphConvolutionNetwork):
        outputs = GraphConvolutionNetwork(out_layer.n_node
                                 , out_layer.input_shape[0][-1]
                                 , output_adjacency=out_layer.output_adjcency
                                 , activation=out_layer.activation)(input_r)  # ?

        # Get an upper layer.
        upper_layer = in_node.inbound_layers
        return _get_reversed_outputs(upper_layer, outputs)
    else:
        raise RuntimeError('Layers must be supported in layer reversing.')


def make_autoencoder_with_sym_sc(autoencoder, name=None):
    """Make autoencoder with symmetry skip-connection.
    
    Parameters
    ----------
    autoencoder: Keras model.
        Autoencoder.
    name: String.
        Symmetric skip-connection autoencoder model's name.
    
    Returns
    -------
    Autoencoder model with symmetry skip-connection.
        Keras model.
    """
    
    # Check exception.?
    # TODO

    # Get encoder and decoder.
    inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in autoencoder.inputs]  
    ae_layers = autoencoder.layers  
    for layer in ae_layers:
        if layer.name == 'encoder':
            encoder = layer
        elif layer.name == 'decoder':
            decoder = layer

    # Make encoder and get skip connection tensors.
    skip_connection_tensors = []
    x = inputs[0] #? 
    for layer in encoder.layers:
        if isinstance(layer, InputLayer):
            continue

        x = layer(x)
        if isinstance(layer, (Dense
                              , DenseBatchNormalization
                              , Conv1D
                              , SeparableConv1D
                              , Conv2D
                              , SeparableConv2D
                              , Conv3D)):
            skip_connection_tensors.append(x)
        else:
            raise ValueError(f'The {layer} is not supported.')
    
    # Make decoder with skip-connection.
    skip_connection_tensors.reverse()
    index = 0
    for layer in decoder.layers:
        if isinstance(layer, (Dense
                              , DenseBatchNormalization
                              , UpSampling1D
                              , Conv2DTranspose
                              , Conv3DTranspose)) \
                and index > 0:
            x = Concatenate(axis=-1)([x, skip_connection_tensors[index]])

            if isinstance(layer, Dense):
                x = Dense(layer.output_shape[-1]
                               , activation=layer.activation
                               , use_bias=layer.use_bias)(x)
            elif isinstance(layer, DenseBatchNormalization):
                dense = Dense(layer.dense_1.input_shape[1]
                              , activation=layer.dense_1.activation
                              , use_bias=layer.dense_1.use_bias)
                batchnormalization = BatchNormalization()
                if layer.activation_1 is not None:
                    activation = layer.activation_1
                else:
                    activation = None
                if layer.dropout_1 is not None:
                    dropout = layer.dropout_1
                else:
                    dropout = None
                dense_batchnormalization = DenseBatchNormalization(dense
                                                                   , batchnormalization
                                                                   , activation=activation
                                                                   , dropout=dropout)
                x = dense_batchnormalization(x)
            elif isinstance(layer, UpSampling1D):  # ?
                if layer.strides[0] >= 2:
                    x = UpSampling2D(size=layer.strides[0])(x)
                else:
                    if isinstance(layer, Conv1D):
                        x = Conv1D(layer.output_shape[-1]
                                        , layer.kernel_size
                                        , strides=1
                                        , padding='same'  # ?
                                        , activation=layer.activation
                                        , use_bias=layer.use_bias)(x)  # ?
                    elif isinstance(layer, SeparableConv1D):
                        x = SeparableConv1D(layer.output_shape[-1]
                                                 , layer.kernel_size
                                                 , strides=1
                                                 , padding='same'  # ?
                                                 , activation=layer.activation
                                                 , use_bias=layer.use_bias)(x)  # ?
            elif isinstance(layer, (Conv2D, SeparableConv2D)):
                if layer.strides[0] >= 2 or layer.strides[1] >= 2:
                    x = Conv2DTranspose(layer.output_shape[-1]
                                             , layer.kernel_size
                                             , strides=layer.strides
                                             , padding='same'  # ?
                                             , activation=layer.activation
                                             , use_bias=layer.use_bias)(x)  # ?
                    #x = UpSampling2D(size=layer.strides[0])(x) #?
                else:
                    if isinstance(layer, Conv2D):
                        x = Conv2D(layer.output_shape[-1]
                                        , layer.kernel_size
                                        , strides=1
                                        , padding='same'  # ?
                                        , activation=layer.activation
                                        , use_bias=layer.use_bias)(x)  # ?
                    elif isinstance(layer, SeparableConv2D):
                        x = SeparableConv2D(layer.output_shape[-1]
                                                 , layer.kernel_size
                                                 , strides=1
                                                 , padding='same'  # ?
                                                 , activation=layer.activation
                                                 , use_bias=layer.use_bias)(x)  # ?
            elif isinstance(layer, (Conv3D)):
                x = Conv3DTranspose(layer.output_shape[-1]
                                         , layer.kernel_size
                                         , strides=layer.strides
                                         , padding='same'  # ?
                                         , activation=layer.activation
                                         , use_bias=layer.use_bias)(x)  # ?
                # x = UpSampling3D(size=layer.strides[0])(x) #?

            index +=1
        elif isinstance(layer, (Dense
                              , DenseBatchNormalization
                              , UpSampling1D
                              , Conv2DTranspose
                              , Conv3DTranspose)) \
                and index == 0:
            if isinstance(layer, Dense):
                x = Dense(layer.output_shape[-1]
                               , activation=layer.activation
                               , use_bias=layer.use_bias)(x)
            elif isinstance(layer, DenseBatchNormalization):
                dense = Dense(layer.dense_1.input_shape[1]
                              , activation=layer.dense_1.activation
                              , use_bias=layer.dense_1.use_bias)
                batchnormalization = BatchNormalization()
                if layer.activation_1 is not None:
                    activation = layer.activation_1
                else:
                    activation = None
                if layer.dropout_1 is not None:
                    dropout = layer.dropout_1
                else:
                    dropout = None
                dense_batchnormalization = DenseBatchNormalization(dense
                                                                   , batchnormalization
                                                                   , activation=activation
                                                                   , dropout=dropout)
                x = dense_batchnormalization(x)
            elif isinstance(layer, UpSampling1D):  # ?
                if layer.strides[0] >= 2:
                    x = UpSampling2D(size=layer.strides[0])(x)
                else:
                    if isinstance(layer, Conv1D):
                        x = Conv1D(layer.output_shape[-1]
                                        , layer.kernel_size
                                        , strides=1
                                        , padding='same'  # ?
                                        , activation=layer.activation
                                        , use_bias=layer.use_bias)(x)  # ?
                    elif isinstance(layer, SeparableConv1D):
                        x = SeparableConv1D(layer.output_shape[-1]
                                                 , layer.kernel_size
                                                 , strides=1
                                                 , padding='same'  # ?
                                                 , activation=layer.activation
                                                 , use_bias=layer.use_bias)(x)  # ?
            elif isinstance(layer, (Conv2D, SeparableConv2D)):
                if layer.strides[0] >= 2 or layer.strides[1] >= 2:
                    x = Conv2DTranspose(layer.output_shape[-1]
                                             , layer.kernel_size
                                             , strides=layer.strides
                                             , padding='same'  # ?
                                             , activation=layer.activation
                                             , use_bias=layer.use_bias)(x)  # ?
                    #x = UpSampling2D(size=layer.strides[0])(x) #?
                else:
                    if isinstance(layer, Conv2D):
                        x = Conv2D(layer.output_shape[-1]
                                        , layer.kernel_size
                                        , strides=1
                                        , padding='same'  # ?
                                        , activation=layer.activation
                                        , use_bias=layer.use_bias)(x)  # ?
                    elif isinstance(layer, SeparableConv2D):
                        x = SeparableConv2D(layer.output_shape[-1]
                                                 , layer.kernel_size
                                                 , strides=1
                                                 , padding='same'  # ?
                                                 , activation=layer.activation
                                                 , use_bias=layer.use_bias)(x)  # ?
            elif isinstance(layer, (Conv3D)):
                x = Conv3DTranspose(layer.output_shape[-1]
                                         , layer.kernel_size
                                         , strides=layer.strides
                                         , padding='same'  # ?
                                         , activation=layer.activation
                                         , use_bias=layer.use_bias)(x)  # ?
                # x = UpSampling3D(size=layer.strides[0])(x) #?

            index +=1
        elif isinstance(layer, InputLayer):
            continue
        else:
            raise ValueError(f'The {layer} is not supported.')
    
    output = x
    return Model(inputs=inputs, outputs=[output], name=name) #?


def make_decoder_from_encoder(encoder, name=None):
    """Make decoder from encoder.

    Parameters
    ----------
    encoder: Keras model
        Encoder.
    name: String.
        Decoder model's name.

    Returns
    -------
    Decoder model
        Keras model.
    """

    # Check exception.?
    # TODO

    # Get a reverse model.
    encoder._init_set_name('encoder')
    decoder = reverse_model(encoder)
    decoder._init_set_name('decoder')

    return decoder


def make_autoencoder_from_encoder(encoder, name=None):
    """Make autoencoder from encoder.
    
    Parameters
    ----------
    encoder: Keras model
        Encoder.
    name: String.
        Autoencoder model's name.
    
    Returns
    -------
    Autoencoder model
        Keras model.
    """
    
    # Check exception.?
    # TODO

    # Get a reverse model.
    encoder._init_set_name('encoder')
    decoder = reverse_model(encoder)
    decoder._init_set_name('decoder')
    
    inputs = [tf.keras.Input(shape=K.int_shape(t)[1:], dtype=t.dtype) for t in encoder.inputs]  
    latents = encoder(inputs)
    output = decoder(latents)    
    return Model(inputs=inputs, outputs=[output], name=name)