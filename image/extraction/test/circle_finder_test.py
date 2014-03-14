#!/usr/bin/python

import numpy as np
import unittest

from image.extraction import CircleFinder
from skimage.data import coins


class CircleFinderTest(unittest.TestCase):
    
    def testBasic(self):
        radii = np.arange(15, 50, 2)
        finder = CircleFinder(coins(), radii)
        finder.find_circles()
        
        # Access a mask to force the mask generation code to run.
        pm = finder.perim_mask


if __name__ == '__main__':
    unittest.main()