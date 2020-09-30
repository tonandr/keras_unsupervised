"""
Created on 2020. 9. 16.

@author: Inwoo Chung (gutomitai@gmail.com)
License: BSD 3 clause.
"""

import numpy as np
import json
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.applications import Xception
import tensorboard

from ku.applications_ext import NobodyConvNet2D


class MNISTDigitClassifier(Model):
    def __init__(self, conf):
        super(MNISTDigitClassifier, self).__init__()

        # Design layers.
        input_shape = (28, 28, 1)

        self.nobody_convnet2d = NobodyConvNet2D(conf, input_shape)
        self.xception = Xception(include_top=False, weights=None)
        self.flatten = Flatten()
        self.dense = Dense(10)

    def call(self, inputs):
        #x = inputs
        x = self.nobody_convnet2d(inputs)
        #x = self.xception(inputs)
        x = self.flatten(x)
        outputs = self.dense(x)

        return outputs


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255, x_test / 255
#x_train = x_train[..., tf.newaxis].astype('float32')[:100]
x_train = x_train[..., tf.newaxis].astype('float32')
#y_train = y_train[:100]
x_test = x_test[..., tf.newaxis].astype('float32')

#x_train = np.tile(x_train, (1, 1, 3))
#x_test = np.tile(x_test, (1, 1, 3))

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(1024).batch(32)


def main():
    with open("mnist_digit_classifier_conf.json", 'r') as f:
        conf = json.load(f)

    #stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    #logdir = 'logs/func/%s' % stamp
    #writer = tf.summary.create_file_writer(logdir)
    model = MNISTDigitClassifier(conf)

    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = optimizers.Adam(lr=conf['hps']['lr']
                          , beta_1=conf['hps']['beta_1']
                          , beta_2=conf['hps']['beta_2']
                          , decay=conf['hps']['decay'])

    train_loss = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    test_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    input_shape = (32, 28, 28, 1)
    model.compile(optimizer=opt, loss=loss_obj, metrics=[train_accuracy], run_eagerly=False)
    #model.build(input_shape)

    #model.summary()
    #model.nobody_convnet2d.summary()

    @tf.function
    def train_step(image, label):
        with tf.GradientTape() as tape:
            pred = model(image, training=True)
            loss = loss_obj(label, pred)

        grad = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grad, model.trainable_variables))

        train_loss(loss)
        train_accuracy(label, pred)

    @tf.function
    def test_step(image, label):
        pred = model(image, training=False)
        loss = loss_obj(label, pred)

        test_loss(loss)
        test_accuracy(label, pred)

    #@tf.function
    #def pred():
    #    pred = model(tf.convert_to_tensor(np.random.rand(32, 28, 28, 1)))

    #logdir = './' #"logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = tf.keras.callbacks.TensorBoard() #log_dir=logdir)
    model.fit(train_dataset, epochs=conf['hps']['epochs']) #, callbacks=[tensorboard_callback])

    #tf.summary.trace_on(graph=True, profiler=True)

    #pred()

    '''
    for e in range(conf['hps']['epochs']):
        train_loss.reset_states()
        train_accuracy.reset_states()
        train_loss.reset_states()
        train_accuracy.reset_states()

        # Train.
        count = 0
        for image, label in train_dataset:
            train_step(image, label)
            count +=1
            if count == 1:
                break

        # Test.
        #for image, label in test_dataset:
        #    test_step(image, label)

        print(f'Epoch {e}'
              f', loss: {train_loss.result()}'
              f', accuracy: {train_accuracy.result()}'
              f', test loss: {test_loss.result()}'
              f', test accuracy: {test_accuracy.result()}')
    '''

    #with writer.as_default():
    #    tf.summary.trace_export(name='mnist', step=0, profiler_outdir=logdir)

if __name__ == '__main__':
    main()