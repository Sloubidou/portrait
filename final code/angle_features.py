#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 18:38:12 2017

@author: slou
"""
import math as mt

#### Location of the face ####

#### Regle des trois traits ####

def eye_line_angle(r_eye_x, r_eye_y, l_eye_x, l_eye_y):     
    """
    compute the angle between the eye line and the abscisse
    """
    A = mt.hypot(l_eye_x - r_eye_x, l_eye_y - r_eye_y)
    B = abs(1 - l_eye_x)
    return mt.degrees(mt.acos(A/B))
    
    


    
