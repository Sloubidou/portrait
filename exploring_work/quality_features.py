#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:37:06 2017

@author: paulinenicolas
"""

import cv2

def blurry(image) :
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm


########### Sharpness function ###########