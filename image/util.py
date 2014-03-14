#!/usr/bin/python

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