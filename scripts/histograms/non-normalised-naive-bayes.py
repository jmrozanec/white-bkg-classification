#How to extract histogram: http://stackoverflow.com/questions/22159160/python-calculate-histogram-of-image
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_preloader
import cv2
from skimage.color import rgb2gray
from skimage import exposure
import skimage.io as io
import numpy as np
import collections

train_file = '../../images/sampling/train-imgs.txt'
test_file = '../../images/sampling/test-imgs.txt'

Dataset = collections.namedtuple('Dataset', ['data', 'target'])

def loaddataset(train_file):
	n_samples = sum(1 for line in open(train_file))
	with open(train_file, "r") as ins:
	    n_features = 255
	    data = np.empty((n_samples, n_features))
	    target = np.empty((n_samples,), dtype=np.int)
	    i = 0
	    for line in ins:
		line = line.rstrip().split()
		filename = line[0]
	        im = io.imread(filename)
		img_gray = rgb2gray(im)
		counts, bins = np.histogram(img_gray, range(256), density=True)
	        data[i] = np.asarray(counts, dtype=np.float64)
	        target[i] = np.asarray(line[1], dtype=np.int)
		i = i+1
	    return Dataset(data=data, target=target)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import BayesianGaussianMixture
from sklearn.linear_model import LogisticRegression

dataset = loaddataset(train_file)
testset = loaddataset(test_file)

gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()
names=['gnb','mnb','bnb']
classifiers=[gnb,mnb,bnb]


for name, clf in zip(names, classifiers):
	scores = cross_val_score(clf, dataset.data, dataset.target, cv=5)
	print("%s, accuracy: %0.4f (+/- %0.4f)" % (name, scores.mean(), scores.std()))

