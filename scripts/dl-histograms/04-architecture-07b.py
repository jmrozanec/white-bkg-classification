#TFLearn bug regarding image loading: https://github.com/tflearn/tflearn/issues/180
#Monochromes img-magick: https://poizan.dk/blog/2014/02/28/monochrome-images-in-imagemagick/
#How to persist a model: https://github.com/tflearn/tflearn/blob/master/examples/basics/weights_persistence.py
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_preloader

train_file = '../../images/sampling/dataset-splits/train-cv-1.txt'
test_file = '../../images/sampling/dataset-splits/test-cv-1.txt'

channels=1
width=64
height=50

X, Y = image_preloader(train_file, image_shape=(height, width),   mode='file', categorical_labels=True,   normalize=True)
testX, testY = image_preloader(test_file, image_shape=(height, width),   mode='file', categorical_labels=True,   normalize=True)

#X = X.reshape([-1, 256, 256, 1])
#testX = testX.reshape([-1, 256, 256, 1])

network = input_data(shape=[None, width, height], name='input')
network = tflearn.layers.core.reshape(network, [-1, width, height, 1], name='Reshape')

network = conv_2d(network, 64, 1, activation='relu')
network = conv_2d(network, 64, 1, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 1, activation='relu')
network = conv_2d(network, 64, 1, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 1, activation='relu')
network = conv_2d(network, 64, 1, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 1, activation='relu')
network = conv_2d(network, 64, 1, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 1, activation='relu')
network = conv_2d(network, 64, 1, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 2, activation='softmax')
# Build neural network and train
network = regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='target')
model = tflearn.DNN(network, tensorboard_verbose=0)

model.fit(X, Y, n_epoch=5, validation_set=(testX, testY), snapshot_step=10, snapshot_epoch=False, show_metric=True, run_id='white-bkg-7')
#epoch=4 => 98%
