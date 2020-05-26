# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:14:59 2019

@author: shefaasaied
"""
import os
import cv2
from skimage import feature

def LocalBinaryPattern(image, numPoints=24, radius=8):
		# Compute the Local Binary Pattern representation of the image, 
        # and then use the LBP representation to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, numPoints,radius, method="default")		
		return lbp.astype("uint8")

parent_dir = "NORMAL"   #### full file path
i=0
for subdir, dirs, files in os.walk(parent_dir):      
    for file in files:
        template = os.path.join(subdir,file)
        img = cv2.imread(os.path.join(subdir, file),0)  
        # Median-filter
        median = cv2.medianBlur(img,5)
        # CLAHE-filter
        clahe = cv2.createCLAHE(clipLimit = 4.0, tileGridSize = (8,8))
        cll = clahe.apply(median)
        # LBP image
        lbp_image = LocalBinaryPattern(cll)
        # ReScaling
        resized_img = cv2.resize(lbp_image, (256, 256), interpolation = cv2.INTER_AREA)
        # Save the new image in the same directory
        cv2.imwrite(template,resized_img)
        i=i+1
        print(i)
