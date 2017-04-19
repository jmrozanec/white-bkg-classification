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

train_file = '../images/sampling/train-imgs.txt'
test_file = '../images/sampling/test-imgs.txt'

Dataset = collections.namedtuple('Dataset', ['data', 'target'], verbose=True)

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
		counts, bins = np.histogram(img_gray, range(256))
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
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import BayesianGaussianMixture
from sklearn.linear_model import LogisticRegression

dataset = loaddataset(train_file)
testset = loaddataset(test_file)

names = ["Nearest Neighbors", "RBF SVM",
         "Decision Tree", "Random Forest", "AdaBoost",
         ]

ab=AdaBoostClassifier(random_state=1)
bgm=BayesianGaussianMixture(random_state=1)
dt=DecisionTreeClassifier(random_state=1)
gb=GradientBoostingClassifier(random_state=1)
lr=LogisticRegression(random_state=1)
rf=RandomForestClassifier(random_state=1)

classifiers = [
    KNeighborsClassifier(3),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    RandomForestClassifier(random_state=1),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
	]

svcl=LinearSVC(random_state=1)
svcg=SVC(random_state=1)
gnb=GaussianNB(2)
qda=QuadraticDiscriminantAnalysis(2)

names=['svcl','lr']
classifiers=[svcl,lr]
params = [
         {'C':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]},
	 {'C':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}
       ]

names=['gp','nb','qda']
classifiers=[GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True, random_state=1), GaussianNB(), QuadraticDiscriminantAnalysis()]

for name, clf in zip(names, classifiers):
	scores = cross_val_score(clf, dataset.data, dataset.target, cv=5)
	print("%s, accuracy: %0.4f (+/- %0.4f)" % (name, scores.mean(), scores.std() * 2))


ab=AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=0.7, n_estimators=100, random_state=1)
dt=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10, max_features=75, max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=1, splitter='best')
gb=GradientBoostingClassifier(criterion='friedman_mse', init=None, learning_rate=0.1, loss='deviance', max_depth=5, max_features=None, max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, presort='auto', random_state=1, subsample=1.0, verbose=0, warm_start=False)
rf=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=10, max_features=10, max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1, oob_score=False, random_state=1, verbose=0, warm_start=False)

clfs = [('ab',ab),('dt',dt),('gb',gb),('rf',rf)]

##Nearest Neighbors, accuracy: 0.91 (+/- 0.01)
##RBF SVM, accuracy: 0.90 (+/- 0.03)
#Decision Tree, accuracy: 0.91 (+/- 0.02)
#Random Forest, accuracy: 0.91 (+/- 0.02)
#AdaBoost, accuracy: 0.91 (+/- 0.02)

#Naive Bayes, accuracy: 0.61 (+/- 0.03)
#QDA, accuracy: 0.62 (+/- 0.02)
#Linear SVM, accuracy: 0.85 (+/- 0.05)
#Gaussian Process, accuracy: 0.80 (+/- 0.19)
#Neural Net, accuracy: 0.74 (+/- 0.35)
