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

train_file = '../images/sampling/train-imgs.txt'
test_file = '../images/sampling/test-imgs.txt'

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
		counts, bins = np.histogram(img_gray, range(256), density=False)
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

clfs = [
	('ab', ab, {'n_estimators':[10,25,50,75,100],'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}),
	('dt', dt,  {'max_depth':[5,10,25,50,75,100],'max_features':[10,25,50,75]}),
	('gb', gb, {'n_estimators':[10,25,50,75,100],'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],'max_depth':[5,10,25,50,75,100]}),
	('rf', rf, {'n_estimators':[10,25,50,75,100],'max_depth':[5,10,25,50,75,100],'max_features':[10,25,50,75]}),
	('svcl', svcl, {'C':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}),
	('lr', lr, {'C':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]})
	]


for name, clf, param in clfs:
	grid = GridSearchCV(estimator=clf, param_grid=param, cv=5)
	grid = grid.fit(dataset.data, dataset.target)
	# summarize the results of the random parameter search
	print(grid.best_score_)
	print(grid.best_estimator_)
