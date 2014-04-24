#!/usr/bin/python

import csv
import exifread
import numpy as np
import pylab

from dateutil.parser import parse as datetimeparse
from skimage import exposure
from skimage.filter import rank
from skimage.morphology import disk


def boost_contrast_global(image):
    """Returns a new image with contrast boosted globally.
    
    Args:
        image: assumed to be grayscale.
    """
    glob = exposure.equalize_hist(image) * 255
    return exposure.rescale_intensity(image)
    
    
def boost_contrast_local(image, radius=3):
    """Returns a new image with contrast boosted locally.
    
    Args:
        image: assumed to be grayscale.
        radius: the size of the radial neighborhood to use.
    """
    return rank.enhance_contrast(image, disk(radius))


def image_datetime(filename):
    """Returns the DateTime of when the image was taken.
    
    Loads DateTime from EXIF data. Will fail if none is given.
    """
    with open(filename) as f:
        exif_tags = exifread.process_file(f)
        assert 'Image DateTime' in exif_tags
        return datetimeparse(exif_tags['Image DateTime'].values)

def get_duration(image_collection):
    """Calculates the duration of a video in the form of an skimage ImageCollection.
    
    Returns the duration in seconds.
    """
    start_datetime = image_datetime(image_collection.files[0])
    end_datetime = image_datetime(image_collection.files[-1])
    return (end_datetime - start_datetime).total_seconds()
        
    
def load_fiji_bbox(fname):
    """Returns (min_row, min_col, max_row, max_col) tuple from file."""
    with open(fname) as f:
        data = f.read()
        pts = [map(int, l.split()) for l in data.splitlines()]
        pts = np.array(pts)
        min_col, min_row = np.min(pts, axis=0)
        max_col, max_row = np.max(pts, axis=0)
        
        return min_row, min_col, max_row, max_col

def load_fiji_line(fname):
    """Returns (endpoint1, endpoint2) tuple from file."""
    with open(fname) as f:
        reader = csv.reader(f, delimiter='\t')
        # unfortunately, line format is x,y and not row,col
        lines = [(int(float(l[1])), int(float((l[0])))) for l in reader]
        assert len(lines) == 2
        return tuple(lines)

def map_line_to_bbox(line_endpoints, bbox):
    """Assumes the line is inside the bounding box."""
    min_row, min_col, max_row, max_col = bbox
    start, end = line_endpoints
    start_row, start_col = start
    end_row, end_col = end
    assert min_row <= start_row <= max_row
    assert min_row <= end_row <= max_row
    assert min_col <= start_col <= max_col
    assert min_col <= end_col <= max_col
    return (start_row - min_row, start_col - min_col), (end_row - min_row, end_col - min_col)
    
        
def save_image(img, filename, title=None):
    pylab.figure()
    pylab.imshow(img)
    if title:
        pylab.title(title)
    pylab.savefig(filename)
    pylab.close()