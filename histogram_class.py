# -*- coding: utf-8 -*-
"""
Created on Sat May 23 02:32:28 2020

@author: Aya Gamal
"""
from skimage import feature
import numpy as np

class Histogram:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius
	def describe(self, lbp_image, eps=1e-7):
		(hist, _) = np.histogram(lbp_image.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
		# return the histogram of Local Binary Patterns
		return hist