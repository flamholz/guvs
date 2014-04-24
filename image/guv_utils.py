#!/usr/bin/python

import argparse
import exifread
import numpy as np
import pandas as pd

from dateutil.parser import parse as datetimeparse
from image import util
from os import path
from scipy import ndimage
from scipy import signal
from skimage import measure
from skimage import segmentation


def subtract_bg(channel, gmm):
    """Subtracts the background and zeros the channel.
    
    Args:
        channel: raw intensity data from the channel.
        gmm: a gaussian mixture model learned from images in this video.
    
    Returns:
        An image the same shape as channel with all
        the background pixels set to 0.
    """
    predictions = gmm.predict(channel.flatten().tolist())
    bg_val = np.argmin(gmm.means_)
    predictions = predictions.reshape(channel.shape)
    above_bg_locs = np.where(predictions != bg_val)
    bg_subtracted = np.zeros(channel.shape)
    raw_vals = channel[above_bg_locs]
    bg_subtracted[above_bg_locs] = raw_vals - raw_vals.min()  # zero
    return bg_subtracted
    

def to_fold_change(channel_nobg):
    """Assumes that the input channel has already been background-subtracted.
    
    Returns:
        An image with the same size as input where each position is the fold change
        between the value at that position and the mean of the above-background signal.
    """
    pos = np.where(channel_nobg > 0)
    fold_change = channel_nobg.copy()
    values = fold_change[pos]
    fold_change[pos] = values / values.mean()
    return fold_change


def find_largest_region(image):
    """Finds the largest region in an image.
    
    Assumes same values implies the same label.
    
    Returns:
        Region properties for the contiguous region in the image with the largest area.
    """
    labels = ndimage.label(image)[0]
    props = measure.regionprops(labels)
    sorted_regions = sorted(props, key=lambda x: x.area, reverse=True)
    return sorted_regions[0]
    
def calc_linescan_enrichment(channel_nobg, line_start, line_end, width=5):
    """Calculate the enrichment of signal at the highest peak along 
       the scan line relative to the two next-highest peaks.
    
    Currently need the dev version of skimage for the profile_line method.

    Args:
        channel_nobg: background-subtracted channel.
        line_start: the starting point of the scan line.
        line_end: the ending point of the scan line.
        width: the width to scan.
    
    Returns: (enrichment, scan)
    """
    linescan = measure.profile_line(channel_nobg, line_start, line_end, linewidth=width)
    mean_linescan = pd.rolling_mean(linescan, 3)
    peaks = signal.argrelmax(mean_linescan)
    peak_values = sorted(mean_linescan[peaks], reverse=True)
    scan_enrichment = peak_values[0] / (peak_values[1] + peak_values[2])
    return scan_enrichment, mean_linescan
