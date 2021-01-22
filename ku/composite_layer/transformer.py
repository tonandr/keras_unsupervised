from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (Layer
    , Dense
    , Add
    , LayerNormalization
    , Concatenate
    , Dropout)

from ..layer_ext import (MultiHeadAttention
    , SIMILARITY_TYPE_DIFF_ABS
    , SIMILARITY_TYPE_PLAIN
    , SIMILARITY_TYPE_SCALED
    , SIMILARITY_TYPE_GENERAL
    , SIMILARITY_TYPE_ADDITIVE)


class Transformer(Layer):
    """Transformer."""

    def __init__(self
                 , num_head
                 , d_output
                 , dropout_rate
                 , similarity_type=SIMILARITY_TYPE_SCALED
                 , layer_norm_f=True
                 , **kwargs):
        super(Transformer, self).__init__(**kwargs)

        # Check exception.
        if isinstance(num_head, int) != True \
                        or isinstance(d_output, int) != True \
                        or (similarity_type in [SIMILARITY_TYPE_DIFF_ABS
                                        , SIMILARITY_TYPE_PLAIN
                                        , SIMILARITY_TYPE_SCALED
                                        , SIMILARITY_TYPE_GENERAL
                                        , SIMILARITY_TYPE_ADDITIVE]) != True \
                        or num_head < 1 \
                        or d_output < 1\
                        or dropout_rate < 0.0:
            raise ValueError('num_head, d_output, dropout_rate or similarity_type is not valid.')

        self.num_head = num_head
        self.d_output = d_output
        self.dropout_rate = dropout_rate
        self.similarity_type = similarity_type
        self.layer_norm_f = layer_norm_f

        # Design layers.
        self.mh_attention_1 = MultiHeadAttention(num_head, d_output, dropout_rate, similarity_type=similarity_type)
        self.dropout_1 = Dropout(self.dropout_rate)
        self.add_1 = Add()
        if self.layer_norm_f:
            self.layer_norm_1 = LayerNormalization(epsilon=1e-6)

        self.mh_attention_2 = MultiHeadAttention(num_head, d_output, dropout_rate, similarity_type=similarity_type)
        self.dropout_2 = Dropout(self.dropout_rate)
        self.add_2 = Add()
        if self.layer_norm_f:
            self.layer_norm_2 = LayerNormalization(epsilon=1e-6)

        self.dense_3_1 = Dense(int(self.d_output * 4), activation='swish')
        self.dense_3_2 = Dense(self.d_output, activation='linear')
        self.dropout_3 = Dropout(self.dropout_rate)
        self.add_3 = Add()
        if self.layer_norm_f:
            self.layer_norm_3 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=None):
        x = inputs[0]
        m = inputs[1]

        x2 = self.mh_attention_1([x, x, x, m])
        x2 = self.dropout_1(x2, training=training)
        x2 = self.add_1([x, x2])
        if self.layer_norm_f:
            x2 = self.layer_norm_1(x2)

        x3 = self.mh_attention_2([x2, x2, x2, m])
        x3 = self.dropout_1(x3, training=training)
        x3 = self.add_2([x2, x3])
        if self.layer_norm_f:
            x3 = self.layer_norm_2(x3)

        x4 = self.dense_3_1(x3)
        x4 = self.dense_3_2(x4)
        x4 = self.dropout_3(x4, training=training)
        x4 = self.add_3([x3, x4])
        if self.layer_norm_f:
            x4 = self.layer_norm_3(x4)
        outputs = x4

        return outputs

    def get_config(self):
        """Get configuration."""
        config = {'num_head': self.num_head
            , 'd_output': self.d_output
            , 'dropout_rate': self.dropout_rate
            , 'similarity_type': self.similarity_type
            , 'layer_norm_f': self.layer_norm_f}
        base_config = super(Transformer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class InterferedTransformer(Layer):
    """Interfered transformer."""

    def __init__(self
                 , num_head
                 , d_output
                 , dropout_rate
                 , similarity_type=SIMILARITY_TYPE_SCALED
                 , layer_norm_f=True
                 , **kwargs):
        super(InterferedTransformer, self).__init__(**kwargs)

        # Check exception.
        if isinstance(num_head, int) != True \
                        or isinstance(d_output, int) != True \
                        or (similarity_type in [SIMILARITY_TYPE_PLAIN
                                        , SIMILARITY_TYPE_SCALED
                                        , SIMILARITY_TYPE_GENERAL
                                        , SIMILARITY_TYPE_ADDITIVE]) != True \
                        or num_head < 1 \
                        or d_output < 1\
                        or dropout_rate < 0.0:
            raise ValueError('num_head, d_output, dropout_rate or similarity_type is not valid.')

        self.num_head = num_head
        self.d_output = d_output
        self.similarity_type = similarity_type
        self.layer_norm_f = layer_norm_f

        # Design layers.
        if self.layer_norm_f:
            self.layer_norm_embedded = LayerNormalization()

        self.mh_attention_1 = MultiHeadAttention(num_head, d_output, dropout_rate, similarity_type=similarity_type)
        self.add_1 = Add()
        if self.layer_norm_f:
            self.layer_norm_1 = LayerNormalization()

        self.mh_attention_2 = MultiHeadAttention(num_head, d_output, dropout_rate, similarity_type=similarity_type)
        self.add_2 = Add()
        if self.layer_norm_f:
            self.layer_norm_2 = LayerNormalization()
        self.concat_2 = Concatenate(axis=-1)

        self.dense_3_1 = Dense(self.d_output, activation='relu')
        self.dense_3_2 = Dense(self.d_output, activation='linear')
        self.add_3 = Add()
        if layer_norm_f:
            self.layer_norm_3 = LayerNormalization()
        self.dropout_3 = Dropout(self.dropout_rate)

    def call(self, inputs, training=None):
        x = inputs[1]
        m = inputs[2]

        num_seq = K.int_shape(x)[1]
        embedded = tf.tile(tf.expand_dims(inputs[0], axis=1), (1, num_seq, 1))
        if self.layer_norm_f:
            embedded = self.layer_norm_embedded(embedded)

        x2 = self.mh_attention_1([x, x, x, m])
        x2 = self.add_1([x, x2])
        if self.layer_norm_f:
            x2 = self.layer_norm_1(x2)

        x3 = self.mh_attention_2([x2, x2, x2, m])
        x3 = self.add_2([x2, x3])
        if self.layer_norm_f:
            x3 = self.layer_norm_2(x3)
        x3 = self.concat_2([x3, embedded])

        x4 = self.dense_3_1(x3)
        x4 = self.dense_3_2(x4)
        x4 = self.add_3([x3, x4])
        if self.layer_norm_f:
            x4 = self.layer_norm_3(x4)
        x4 = self.dropout_3(x4, training=training)
        outputs = x4

        return outputs

    def get_config(self):
        """Get configuration."""
        config = {'num_head': self.num_head
            , 'd_output': self.d_output
            , 'dropout_rate': self.dropout_rate
            , 'similarity_type': self.similarity_type
            , 'layer_norm_f': self.layer_norm_f}
        base_config = super(InterferedTransformer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
