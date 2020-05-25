# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:18:57 2020

@author: Aya Gamal
"""

from skimage import feature
import cv2
import glob
import os

def LocalBinaryPattern(image, numPoints=24, radius=8):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, numPoints,
			radius, method="uniform")
		
		return lbp 


#test function 
parent_dir = "" #### full file path
i=0
for subdir, dirs, files in os.walk(parent_dir):
    for file in files:
        template = os.path.join(subdir,file)
        print(os.path.join(subdir, file))
        img = cv2.imread(os.path.join(subdir, file),0)
        lbp_image = LocalBinaryPattern(img)
        cv2.imwrite(template,lbp_image)
        i=i+1
        print(i)
