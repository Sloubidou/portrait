#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:37:06 2017

@author: paulinenicolas
"""

from numpy.linalg import norm
from math import hypot

#### Location of the face ####

#### Regle des trois traits ####


def get_face_center(x_0, width_f, y_0, height_f):
    'x_0, width_f, y_0, height_f in percentage'
    
    x_c = x_0 + width_f/2
    y_c = y_0 + height_f/2
    return (x_c, y_c) 

def dist_rule_thirds(x_c, y_c):     
    """
    x_c, y_c position of the center of the face in %
    """
    top_third = 1/3
    bot_third = 1/3
    '''distance between the face center & the 4 points (rule of thirds)'''
    dist_1 = hypot((top_third - y_c),(top_third - x_c))
    dist_2 = hypot((top_third - y_c),(bot_third - x_c))
    dist_3 = hypot((bot_third - y_c),(bot_third - x_c))
    dist_4 = hypot((bot_third - y_c),(top_third - x_c))
    
    '''distance between the face center & the 4 lines (rule of thirds)'''
    
    dist_l1 = norm(top_third - x_c) #vertical 1
    dist_l2 = norm(bot_third - x_c) #vertical 2
    dist_l3 = norm(top_third - y_c) # horizontal 1
    dist_l4 = norm(bot_third - y_c) # horizontal 2
    
    return (dist_1, dist_2, dist_3, dist_4, dist_l1,dist_l2,dist_l3,dist_l4)

def face_ratio(width_f, height_f):
    """
    Ratio of face to image: Approx number of pixels of the face to that in the background
    """
    ratio = width_f * height_f
    return ratio

def eye_position(y_left_eye, y_right_eye):
    """
    Distance between eyes and the top third of the picture
    """
    top_third = 1/3
    dist_eye1 = norm(top_third-y_left_eye)
    dist_eye2 = norm(top_third-y_right_eye)
    return (dist_eye1, dist_eye2) 