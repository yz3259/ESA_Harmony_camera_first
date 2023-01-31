# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt


def app2act_transform(lat,lon,h,theta,phi):
    """
    Input:
        lat,lon are in degrees
        h: hight of clouds in [km]
        theta and phi are in degrees
        
    """
    R = 6.371e6 # the radius of earth unit in meters
    deg = np.pi*13/180 # rad of 13 degrees
    a = h*1000/(np.tan(theta-np.pi/4))
    dx = -np.sin(phi-deg)*a
    dy = -np.cos(phi-deg)*a
    
    lat1 = lat + 180*dx/(np.pi*R*np.cos(lon*np.pi/180))
    lon1 = lon + 180*dy/(np.pi*R)
    
    return lat1,lon1


