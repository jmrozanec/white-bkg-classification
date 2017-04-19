#https://github.com/tflearn/tflearn/issues/180
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_preloader
import skimage
from skimage import data
from skimage import filters
import os
from skimage import io
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

reds="../images/pictures/red/"
greens="../images/pictures/green/"
redshist="../images/histograms/red/"
greenshist="../images/histograms/green/"
directory=reds
histdirectory=redshist
for filename in os.listdir(directory):
    if filename.endswith(".jpg"): 
	img = io.imread(os.path.join(directory, filename))
	hist, bin_edges = np.histogram(img, bins=255)
	bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
	binary_img = img > 0.8
	plt.figure(figsize=(1,1))
	fig, ax = plt.subplots(nrows=1, ncols=1) #http://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it-using-matplotlib-so-it-can-be
	plt.plot(bin_centers, hist, lw=2)
	fig.savefig(os.path.join(histdirectory, filename), bbox_inches='tight')
	plt.close()
    else:
        continue
