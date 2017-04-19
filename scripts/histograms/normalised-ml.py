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
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import BayesianGaussianMixture
from sklearn.linear_model import LogisticRegression

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


dataset = loaddataset(train_file)
testset = loaddataset(test_file)

ab=AdaBoostClassifier(random_state=1)
bgm=BayesianGaussianMixture(random_state=1)
dt=DecisionTreeClassifier(random_state=1)
gb=GradientBoostingClassifier(random_state=1)
lr=LogisticRegression(random_state=1)
rf=RandomForestClassifier(random_state=1)
svcl=LinearSVC(random_state=1)

ab=AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=0.1, n_estimators=10, random_state=1)
dt=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5, max_features=10, max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=1, splitter='best')
gb=GradientBoostingClassifier(criterion='friedman_mse', init=None, learning_rate=0.1, loss='deviance', max_depth=5, max_features=None, max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=25, presort='auto', random_state=1, subsample=1.0, verbose=0, warm_start=False)
rf=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=5, max_features=10, max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=75, n_jobs=1, oob_score=False, random_state=1, verbose=0, warm_start=False)
svcl=LinearSVC(C=0.9, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='squared_hinge', max_iter=1000, multi_class='ovr', penalty='l2', random_state=1, tol=0.0001, verbose=0)
lr=LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2', random_state=1, solver='liblinear', tol=0.0001, verbose=0, warm_start=False)


clfs = [
	('ab', ab),
	('dt', dt),
	('gb', gb),
	('rf', rf),
	('svcl', svcl),
	('lr', lr)
	]


for name, clf in clfs:
	scores = cross_val_score(clf, dataset.data, dataset.target, cv=5)
        print("%s, accuracy: %0.4f (+/- %0.4f)" % (name, scores.mean(), scores.std()))
