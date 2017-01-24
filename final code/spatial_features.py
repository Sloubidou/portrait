#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:37:06 2017

@author: paulinenicolas
"""

import cv2

#### Location of the face ####

#### Regle des trois traits ####

from numpy.linalg import norm

def get_face_center(width_im, height_im, x_0, width_f, y_0, height_f):
    'x_0, width_f, y_0, height_f in percentage'
    'width_im, height_im in number of pixels' 
    
    x_c = int((x_0 + width_f/2) * width_im)
    y_c = int((y_0 + height_f/2) * height_im)
    
    return (x_c, y_c) 

def dist_rule_thirds(width_im, height_im, x_c, y_c): 
    """
    x_c, y_c position of the center of the face in number of pixels
    width_im, height_im in number of pixels
    """
    
    '''distance between the face center & the 4 points (rule of thirds)'''

    hw = width_im * height_im # normalization
    
    dist_1 = norm(((height_im/3 - y_c), (width_im/3-x_c)))/hw
    dist_2 = norm(((height_im/3 - y_c),(2*width_im/3-x_c)))/hw
    dist_3 = norm(((2*height_im/3 - y_c),(2*width_im/3-x_c)))/hw
    dist_4 = norm(((2*height_im/3 - y_c),(width_im/3-x_c)))/hw
    
    '''distance between the face center & the 4 lines (rule of thirds)'''

    dist_l1 = norm(((height_im/3-x_c)))/hw #vertical 1
    dist_l2 = norm(((2*height_im/3)-x_c))/hw #vertical 2
    dist_l3 = norm(((width_im/3-y_c)))/hw # horizontal 1
    dist_l4 = norm(((2*width_im/3-y_c)))/hw # horizontal 2
    
    return (dist_1, dist_2, dist_3, dist_4, dist_l1,dist_l2,dist_l3,dist_l4)

def face_ratio(width_f, height_f):
    """
    Ratio of face to image: Approx number of pixels of the face to that in the background
    """
    ratio = width_f * height_f
    return ratio