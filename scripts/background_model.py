#!/usr/bin/python

import argparse
import numpy as np
import pylab

from image import util
from os import path
from skimage.io import ImageCollection
from skimage.filter import rank
from skimage.morphology import disk

from sklearn import mixture
from sklearn.externals import joblib


def fit_gmm(values):
    gmm = mixture.GMM(n_components=3)
    gmm.fit(values)
    return gmm
    

parser = argparse.ArgumentParser(description='Analyze GUV videos')
parser.add_argument('-b', '--base_directory',
                    default='/Users/flamholz/Documents/Rotations/Fletcher/atto_390_dGFP_4_5_14/videos/scanning_around_12/',
                    help='Location of image and other files.')
args = parser.parse_args()

# Paths to files
base_dir = args.base_directory
bbox_path = path.join(base_dir, 'bbox.txt')
blue_gmm_path = path.join(base_dir, 'blue_gmm.pkl')
green_gmm_path = path.join(base_dir, 'green_gmm.pkl')
blue_channel_path = path.join(base_dir, 'blue_channel_mean/*.tif')
green_channel_path = path.join(base_dir, 'green_channel_mean/*.tif')

blue_collection = ImageCollection(blue_channel_path)
green_collection = ImageCollection(green_channel_path)

bbox = util.load_fiji_bbox(bbox_path)
min_r, min_c, max_r, max_c = bbox

blue_channels = []
green_channels = []
N_TRAINING_IMAGES = 30
print 'Collecting', N_TRAINING_IMAGES, 'training images'

# Establish mixture models for labeling the background using first N images.
# TODO: should we smooth the images? Maybe do that in advance?
# TODO: save the gmms so we don't regenerate them every run. 
for i, (blue_channel, green_channel) in enumerate(zip(blue_collection, green_collection)):
    green_channels.append(green_channel[min_r:max_r, min_c:max_c].flatten())
    blue_channels.append(blue_channel[min_r:max_r, min_c:max_c].flatten())
    
    if i > N_TRAINING_IMAGES:
        break

print 'Building Gaussian mixture models'
blue_values = np.hstack(blue_channels)
green_values = np.hstack(green_channels)
blue_gmm = fit_gmm(blue_values)
green_gmm = fit_gmm(green_values)

print 'Writing mixture models'
joblib.dump(blue_gmm, blue_gmm_path, compress=9)
joblib.dump(green_gmm, green_gmm_path, compress=9)