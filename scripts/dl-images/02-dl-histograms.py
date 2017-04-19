#TFLearn bug regarding image loading: https://github.com/tflearn/tflearn/issues/180
#Monochromes img-magick: https://poizan.dk/blog/2014/02/28/monochrome-images-in-imagemagick/
#How to persist a model: https://github.com/tflearn/tflearn/blob/master/examples/basics/weights_persistence.py
from __future__ import division, print_function, absolute_import

import argparse
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_preloader
from tflearn.layers.merge_ops import merge
import uuid
import numpy as np

def main(args):
	experiment='dl-histograms'
	channels=1
	width=64
	height=64
	num_class=2
	epochs=15
	folds=5
	architecturescount=10
	architectureid=args.architectureid
        fold=args.fold
	test_file='../../images/sampling/lists/histograms/test-images.txt'

#       for architectureid in range(1,architecturescount):
	accuracies=[]
	#for fold in range(1,folds):
	runid='{}-architecture{}-fold{}'.format(experiment,architectureid,fold)
	arch = architecture(architectureid, width, height, channels, num_class)
	train_file = '../../images/sampling/lists/histograms/splits/train-cv-{}.txt'.format(fold)
	validate_file = '../../images/sampling/lists/histograms/splits/validate-cv-{}.txt'.format(fold)

	X, Y = image_preloader(train_file, image_shape=(height, width),   mode='file', categorical_labels=True,   normalize=True)
	valX, valY = image_preloader(validate_file, image_shape=(height, width),   mode='file', categorical_labels=True,   normalize=True)
	testX, testY = image_preloader(test_file, image_shape=(height, width),   mode='file', categorical_labels=True,   normalize=True)

	arch.fit(X, Y, n_epoch=epochs, validation_set=(valX, valY), snapshot_step=10, snapshot_epoch=False, show_metric=True, run_id=runid)
	arch.save('arch-id{}-fold{}.tfl'.format(architectureid, fold))
#       accuracies.append(arch.evaluate(testX, testY)[0])
#       accuracies=np.asarray(accuracies)
	accuracy=arch.evaluate(testX, testY)[0]
	append(experiment, architectureid, fold, accuracy)

def append(experiment, architectureid, fold, accuracy):
#	line='{},[{}],{},{}\n'.format(architectureid,','.join([str(i) for i in accuracies]),accuracies.mean(), accuracies.std())
	line='{},{},{}\n'.format(architectureid,fold, accuracy)
	with open('{}.csv'.format(experiment), "a") as report:
    		report.write(line)

def architecture(id, width, height, channels, num_class):
	"""
	Obtain DNN architecture for given id.
	"""
	input = input_data(shape=[None, width, height], name='input')
	input = tflearn.layers.core.reshape(input, [-1, width, height, channels], name='Reshape')

	if id == 1:
		return architecture01(input, num_class)
	if id == 2:
                return architecture02(input, num_class)
        if id == 3:
                return architecture03(input, num_class)
        if id == 4:
                return architecture04(input, num_class)
        if id == 5:
                return architecture05(input, num_class)
        if id == 6:
                return architecture06(input, num_class)
        if id == 7:
                return architecture07(input, num_class)
        if id == 8:
                return architecture08(input, num_class)
        if id == 9:
                return architecture09(input, num_class)
        if id == 10:
                return architecture10(input, num_class)


def architecture01(input, num_class):
	network = conv_2d(input, 64, 1, activation='relu', regularizer="L2")
	network = batch_normalization(network)
	network = conv_2d(network, 64, 1, activation='relu', regularizer="L2")
	network = max_pool_2d(network, 2)
	network = fully_connected(network, 64, activation='tanh')
	network = dropout(network, 0.8)
	network = fully_connected(network, num_class, activation='softmax')

	network = regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='target')
	return tflearn.DNN(network, tensorboard_verbose=0)

def architecture02(input, num_class):
	network = batch_normalization(input)
	network = conv_2d(network, 64, 1, activation='relu', regularizer="L2")
	network = max_pool_2d(network, 2)
	network = batch_normalization(network)
	network = conv_2d(network, 64, 1, activation='relu', regularizer="L2")
	network = max_pool_2d(network, 2)
	network = batch_normalization(network)
	network = conv_2d(network, 64, 1, activation='relu', regularizer="L2")
	network = max_pool_2d(network, 2)
	network = fully_connected(network, num_class, activation='softmax')

	network = regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='target')
	return tflearn.DNN(network, tensorboard_verbose=0)

#ResNet: Taken and adapted from from https://github.com/tflearn/tflearn/blob/master/examples/images/residual_network_mnist.py
def architecture03(input, num_class):
	net = tflearn.conv_2d(input, 64, 3, activation='relu', bias=False)
	net = tflearn.residual_bottleneck(net, 3, 16, 64)
	net = tflearn.residual_bottleneck(net, 1, 32, 128, downsample=False)
	net = tflearn.residual_bottleneck(net, 2, 32, 128)
	net = tflearn.residual_bottleneck(net, 1, 64, 256, downsample=False)
	net = tflearn.residual_bottleneck(net, 2, 64, 256)
	net = tflearn.batch_normalization(net)
	net = tflearn.activation(net, 'relu')
	net = tflearn.global_avg_pool(net)
	net = tflearn.fully_connected(net, num_class, activation='softmax')

	net = tflearn.regression(net, optimizer='momentum', loss='categorical_crossentropy', learning_rate=0.001)
	return tflearn.DNN(net, tensorboard_verbose=0)

def architecture04(input, num_class):
	network = conv_2d(input, 64, 1, activation='relu', regularizer="L2")
	network = batch_normalization(network)
	network = conv_2d(network, 64, 1, activation='relu', regularizer="L2")
	network = batch_normalization(network)
	network = conv_2d(network, 64, 1, activation='relu', regularizer="L2")
	network = max_pool_2d(network, 2)
	network = fully_connected(network, 64, activation='tanh')
	network = dropout(network, 0.8)
	network = fully_connected(network, num_class, activation='softmax')
	
	network = regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='target')
	return tflearn.DNN(network, tensorboard_verbose=0)

def architecture05(input, num_class):
	network = conv_2d(input, 128, 1, activation='relu', regularizer="L2")
	network = batch_normalization(network)
	network = fully_connected(network, 64, activation='tanh')
	network = dropout(network, 0.8)
	network = fully_connected(network, num_class, activation='softmax')

	network = regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='target')
	return tflearn.DNN(network, tensorboard_verbose=0)

def architecture06(input, num_class):
	network1 = batch_normalization(conv_2d(input, 64, 1, activation='relu', regularizer="L2"))
	network2 = batch_normalization(conv_2d(input, 64, 3, activation='relu', regularizer="L2"))
	network = merge([network1, network2],'concat')
	network = fully_connected(network, 64, activation='tanh')
	network = dropout(network, 0.8)
	network = fully_connected(network, num_class, activation='softmax')
	
	network = regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='target')
	return tflearn.DNN(network, tensorboard_verbose=0)

def architecture07(input, num_class):
	network = conv_2d(input, 64, 1, activation='relu')
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
	network = fully_connected(network, num_class, activation='softmax')
	
	network = regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='target')
	return tflearn.DNN(network, tensorboard_verbose=0)

def architecture08(input, num_class):
	network = conv_2d(input, 64, 1, activation='relu', regularizer="L2")
	network = batch_normalization(network)
	network = max_pool_2d(network, 2)
	network = fully_connected(network, 64, activation='tanh')
	network = dropout(network, 0.8)
	network = fully_connected(network, num_class, activation='softmax')
	
	network = regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='target')
	return tflearn.DNN(network, tensorboard_verbose=0)

#VGG16 implementation. Taken from: https://github.com/tflearn/tflearn/blob/master/examples/images/vgg_network_finetuning.py
def architecture09(input, num_class):
	x = tflearn.conv_2d(input, 64, 3, activation='relu', scope='conv1_1')
	x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2')
	x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

	x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1')
	x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2')
	x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

	x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1')
	x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2')
	x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3')
	x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')

	x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1')
	x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2')
	x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3')
	x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')

	x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1')
	x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2')
	x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3')
	x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

	x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc6')
	x = tflearn.dropout(x, 0.5, name='dropout1')

	x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
	x = tflearn.dropout(x, 0.5, name='dropout2')

	x = tflearn.fully_connected(x, num_class, activation='softmax', scope='fc8', restore=False)

	x = regression(x, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='target')
	return tflearn.DNN(x, tensorboard_verbose=0)

#GoogleLeNet: Taken and adapted from https://github.com/tflearn/tflearn/blob/master/examples/images/googlenet.py
def architecture10(input, num_class):
	conv1_7_7 = conv_2d(input, 64, 7, strides=2, activation='relu', name = 'conv1_7_7_s2')
	pool1_3_3 = max_pool_2d(conv1_7_7, 3,strides=2)
	pool1_3_3 = local_response_normalization(pool1_3_3)
	conv2_3_3_reduce = conv_2d(pool1_3_3, 64,1, activation='relu',name = 'conv2_3_3_reduce')
	conv2_3_3 = conv_2d(conv2_3_3_reduce, 192,3, activation='relu', name='conv2_3_3')
	conv2_3_3 = local_response_normalization(conv2_3_3)
	pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')
	inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
	inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96,1, activation='relu', name='inception_3a_3_3_reduce')
	inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128,filter_size=3,  activation='relu', name = 'inception_3a_3_3')
	inception_3a_5_5_reduce = conv_2d(pool2_3_3,16, filter_size=1,activation='relu', name ='inception_3a_5_5_reduce' )
	inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name= 'inception_3a_5_5')
	inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
	inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')

	# merge the inception_3a__
	inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

	inception_3b_1_1 = conv_2d(inception_3a_output, 128,filter_size=1,activation='relu', name= 'inception_3b_1_1' )
	inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
	inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3,  activation='relu',name='inception_3b_3_3')
	inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu', name = 'inception_3b_5_5_reduce')
	inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5,  name = 'inception_3b_5_5')
	inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
	inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1,activation='relu', name='inception_3b_pool_1_1')

	#merge the inception_3b_*
	inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat',axis=3,name='inception_3b_output')

	pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')
	inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
	inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
	inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', name='inception_4a_3_3')
#	inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
#	inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', name='inception_4a_5_5')
	inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
	inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')

	inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')


	inception_4b_1_1 = conv_2d(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
	inception_4b_3_3_reduce = conv_2d(inception_4a_output, 112, filter_size=1, activation='relu', name='inception_4b_3_3_reduce')
	inception_4b_3_3 = conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu', name='inception_4b_3_3')
#	inception_4b_5_5_reduce = conv_2d(inception_4a_output, 24, filter_size=1, activation='relu', name='inception_4b_5_5_reduce')
#	inception_4b_5_5 = conv_2d(inception_4b_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4b_5_5')

	inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1,  name='inception_4b_pool')
	inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, filter_size=1, activation='relu', name='inception_4b_pool_1_1')

	inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_pool_1_1], mode='concat', axis=3, name='inception_4b_output')


	inception_4c_1_1 = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu',name='inception_4c_1_1')
	inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_3_3_reduce')
	inception_4c_3_3 = conv_2d(inception_4c_3_3_reduce, 256,  filter_size=3, activation='relu', name='inception_4c_3_3')
#	inception_4c_5_5_reduce = conv_2d(inception_4b_output, 24, filter_size=1, activation='relu', name='inception_4c_5_5_reduce')
#	inception_4c_5_5 = conv_2d(inception_4c_5_5_reduce, 64,  filter_size=5, activation='relu', name='inception_4c_5_5')

	inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
	inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, filter_size=1, activation='relu', name='inception_4c_pool_1_1')

	inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_pool_1_1], mode='concat', axis=3,name='inception_4c_output')

	inception_4d_1_1 = conv_2d(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
	inception_4d_3_3_reduce = conv_2d(inception_4c_output, 144, filter_size=1, activation='relu', name='inception_4d_3_3_reduce')
	inception_4d_3_3 = conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu', name='inception_4d_3_3')
#	inception_4d_5_5_reduce = conv_2d(inception_4c_output, 32, filter_size=1, activation='relu', name='inception_4d_5_5_reduce')
#	inception_4d_5_5 = conv_2d(inception_4d_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4d_5_5')
	inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1,  name='inception_4d_pool')
	inception_4d_pool_1_1 = conv_2d(inception_4d_pool, 64, filter_size=1, activation='relu', name='inception_4d_pool_1_1')

	inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_pool_1_1], mode='concat', axis=3, name='inception_4d_output')

	inception_4e_1_1 = conv_2d(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
	inception_4e_3_3_reduce = conv_2d(inception_4d_output, 160, filter_size=1, activation='relu', name='inception_4e_3_3_reduce')
	inception_4e_3_3 = conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_4e_3_3')
#	inception_4e_5_5_reduce = conv_2d(inception_4d_output, 32, filter_size=1, activation='relu', name='inception_4e_5_5_reduce')
#	inception_4e_5_5 = conv_2d(inception_4e_5_5_reduce, 128,  filter_size=5, activation='relu', name='inception_4e_5_5')
	inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1,  name='inception_4e_pool')
	inception_4e_pool_1_1 = conv_2d(inception_4e_pool, 128, filter_size=1, activation='relu', name='inception_4e_pool_1_1')


	inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_pool_1_1],axis=3, mode='concat')

	pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')


#	inception_5a_1_1 = conv_2d(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
#	inception_5a_3_3_reduce = conv_2d(pool4_3_3, 160, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
#	inception_5a_3_3 = conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_5a_3_3')
#	inception_5a_5_5_reduce = conv_2d(pool4_3_3, 32, filter_size=1, activation='relu', name='inception_5a_5_5_reduce')
#	inception_5a_5_5 = conv_2d(inception_5a_5_5_reduce, 128, filter_size=5,  activation='relu', name='inception_5a_5_5')
#	inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1,  name='inception_5a_pool')
#	inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 128, filter_size=1,activation='relu', name='inception_5a_pool_1_1')

#	inception_5a_output = merge([inception_5a_1_1, inception_5a_pool_1_1], axis=3,mode='concat')


#	inception_5b_1_1 = conv_2d(inception_5a_output, 384, filter_size=1,activation='relu', name='inception_5b_1_1')
#	inception_5b_3_3_reduce = conv_2d(inception_5a_output, 192, filter_size=1, activation='relu', name='inception_5b_3_3_reduce')
#	inception_5b_3_3 = conv_2d(inception_5b_3_3_reduce, 384,  filter_size=3,activation='relu', name='inception_5b_3_3')
#	inception_5b_5_5_reduce = conv_2d(inception_5a_output, 48, filter_size=1, activation='relu', name='inception_5b_5_5_reduce')
#	inception_5b_5_5 = conv_2d(inception_5b_5_5_reduce,128, filter_size=5,  activation='relu', name='inception_5b_5_5' )
#	inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1,  name='inception_5b_pool')
#	inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 128, filter_size=1, activation='relu', name='inception_5b_pool_1_1')
#	inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_pool_1_1], axis=3, mode='concat')

#	pool5_7_7 = avg_pool_2d(inception_4e_output, kernel_size=7, strides=1)
	pool5_7_7 = dropout(inception_4e_output, 0.4)
	loss = fully_connected(pool5_7_7, num_class, activation='softmax')
	network = regression(loss, optimizer='momentum', loss='categorical_crossentropy', learning_rate=0.001)
	return tflearn.DNN(network, checkpoint_path='model_googlenet', max_checkpoints=1, tensorboard_verbose=0)


parser = argparse.ArgumentParser()
parser.add_argument('-a', '--architectureid', type=int)
parser.add_argument('-f', '--fold', type=int)
args = parser.parse_args()

main(args)
