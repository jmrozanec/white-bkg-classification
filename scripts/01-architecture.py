#https://github.com/tflearn/tflearn/issues/180
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_preloader

train_file = '../images/sampling/train.txt'
test_file = '../images/sampling/test.txt'

X, Y = image_preloader(train_file, image_shape=(256, 256),   mode='file', categorical_labels=True,   normalize=True)
testX, testY = image_preloader(test_file, image_shape=(256, 256),   mode='file', categorical_labels=True,   normalize=True)

network = input_data(shape=[None, 256, 256], name='input')
network = tflearn.layers.core.reshape(network, [-1, 256, 256, 1], name='Reshape')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 2, activation='softmax')
# Build neural network and train
network = regression(network, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='target')
model = tflearn.DNN(network, tensorboard_verbose=3)
model.fit(X, Y, n_epoch=5, validation_set=(testX, testY), snapshot_step=100, snapshot_epoch=False, show_metric=True, run_id='white-bkg')
