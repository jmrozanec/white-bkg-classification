#TFLearn bug regarding image loading: https://github.com/tflearn/tflearn/issues/180
#Monochromes img-magick: https://poizan.dk/blog/2014/02/28/monochrome-images-in-imagemagick/
#How to persist a model: https://github.com/tflearn/tflearn/blob/master/examples/basics/weights_persistence.py
from __future__ import division, print_function, absolute_import
from PIL import Image
import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_preloader

train_file = './../images/sampling/train-imgs.txt'
test_file = '../../images/sampling/test-imgs.txt'

channels=3
width=64
height=50

X, Y = image_preloader(train_file, image_shape=(height, width),   mode='file', categorical_labels=True,   normalize=True)
testX, testY = image_preloader(test_file, image_shape=(height, width, channels),   mode='file', categorical_labels=True,   normalize=True)

network = input_data(shape=[None, width, height, channels], name='input')
network = tflearn.layers.core.reshape(network, [-1, width, height, channels], name='Reshape')

network = conv_2d(network, 64, 1, activation='relu', regularizer="L2")
network = batch_normalization(network)
network = conv_2d(network, 64, 1, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = fully_connected(network, 64, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 2, activation='softmax')
# Build neural network and train
network = regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='target')
model = tflearn.DNN(network, tensorboard_verbose=0)

model.fit(X, Y, n_epoch=5, validation_set=0.2, snapshot_step=10, snapshot_epoch=False, show_metric=True, run_id='white-bkg-1')
result = model.evaluate(testX, testY)
print(result)
#epoch=4 => 98%
