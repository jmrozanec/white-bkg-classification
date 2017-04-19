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

train_file = '../../images/sampling/train.txt'
test_file = '../../images/sampling/test.txt'

channels=1
width=64
height=50

X, Y = image_preloader(train_file, image_shape=(height, width),   mode='file', categorical_labels=True,   normalize=True)
#valX, valY = image_preloader(validate_file, image_shape=(height, width),   mode='file', categorical_labels=True,   normalize=True)
testX, testY = image_preloader(test_file, image_shape=(height, width),   mode='file', categorical_labels=True,   normalize=True)

net = input_data(shape=[None, width, height], name='input')
net = tflearn.layers.core.reshape(net, [-1, width, height, 1], name='Reshape')

net = tflearn.conv_2d(net, 64, 3, activation='relu', bias=False)
# Residual blocks
net = tflearn.residual_bottleneck(net, 3, 16, 64)
net = tflearn.residual_bottleneck(net, 1, 32, 64, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 32, 64)
net = tflearn.residual_bottleneck(net, 1, 64, 64, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 64, 64)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
# Regression
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='momentum', loss='categorical_crossentropy', learning_rate=0.1)
# Training
model = tflearn.DNN(net, tensorboard_verbose=0)

model.fit(X, Y, n_epoch=5, validation_set=0.2, snapshot_step=10, snapshot_epoch=False, show_metric=True, run_id='white-bkg-1')
result = model.evaluate(testX, testY)
print(result)
