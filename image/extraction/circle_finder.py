#!/usr/bin/python

import numpy as np

from skimage import filter as skfilter
from skimage.draw import circle_perimeter
from skimage.feature import peak_local_max
from skimage.transform import hough_circle


class CircleFinder(object):
    """Class that finds circles in greyscale images."""
    
    def __init__(self, image, radii, n_circles=5):
        self.image = image
        self.radii_to_find = radii
        self.n_circles = n_circles
        self.hough_res = None
        
        self.centers = []
        self.accums = []
        self.radii_found = []
        
        self._center_mask = None
        self._perim_mask = None

    def find_circles(self):        
        """Uses a Hough transform to find circular shapes in the image.
            
        Initializes state variables appropriately.
        """
        edges = skfilter.canny(self.image, sigma=3, low_threshold=10, high_threshold=50)
        self.hough_res = hough_circle(edges, self.radii_to_find)
        
        for radius, h in zip(self.radii_to_find, self.hough_res):
            peaks = peak_local_max(h, num_peaks=2)
            self.centers.extend(peaks)
            self.accums.extend(h[peaks[:, 0], peaks[:, 1]])
            self.radii_found.extend([radius, radius])
        
    def _init_masks(self):
        """Initializes the masks on demand."""
        shape = self.image.shape
        # perimeter mask is color.
        self._perim_mask = np.zeros((shape[0], shape[1], 3))
        self._center_mask = np.zeros(shape) 
        # Draw the most prominent 5 circles
        for idx in np.argsort(self.accums)[::-1][:self.n_circles]:
            center_x, center_y = self.centers[idx]
            radius = self.radii_found[idx]
            cx, cy = circle_perimeter(center_y, center_x, radius)
            in_frame_idx = np.where(np.logical_and(cx <= shape[0], cy <= shape[1]))
            self._center_mask[center_x, center_y] = 1
            self._perim_mask[cy[in_frame_idx], cx[in_frame_idx]] = (220, 20, 20)
    
    @property
    def perim_mask(self):
        if self._perim_mask is None:
            self._init_masks()
        return self._perim_mask
    
    @property
    def center_mask(self):
        if self._center_mask is None:
            self._init_masks()
        return self._center_mask