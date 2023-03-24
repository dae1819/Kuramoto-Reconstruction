# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:33:54 2023

@author: Dan
"""

"""
This code does analysis on the thresholding scheme, 
buit is work in progress

The function requires libraries called RegscorePy and kneed

"""




from kneed import KneeLocator
from RegscorePy import aic
import numpy as np
from scipy.ndimage import gaussian_filter

def S(SC,thresh,pairs):
    S=0
    sorted_connections = {k: v for k, v in sorted(pairs.items(), key=lambda item: item[1])}
    
    for i,P2 in enumerate(list(sorted_connections.values())):
        key = list(sorted_connections.keys())[i]
    
        if frozenset((key)) in SC.edges.members():
            if P2 < thresh:
                S += 1
                
        else:
            if P2 > thresh:
                S += 1
                
    return S



def thresh_aic(SC,num,pairs):
    sorted_connections = {k: v for k, v in sorted(pairs.items(), key=lambda item: item[1])}
    
    if num >= len(list(sorted_connections.values())):
        raise Exception("num cannot be greater than or equal to the number of probabilities") 
    
    y = np.zeros(len(list(sorted_connections.values())))
    for i,key in enumerate(list(sorted_connections.keys())):
        if frozenset((key)) in SC.edges.members():
            y[i] = 1
        else:
            y[i] = 0
            
    y_pred = np.zeros(len(list(sorted_connections.values())))
    for i,P2 in enumerate(list(sorted_connections.values())):
        key = list(sorted_connections.keys())[i]
        if P2 > np.max(list(sorted_connections.values())[:-1*num]):
                y_pred[i] = 1
        else:
                y_pred[i] = 0
    
    print(num)
    return aic.aic(y,y_pred,int(num))
     

def thresh_knee(sorted_connections):
    kneedle = KneeLocator(np.arange(len(sorted_connections)), list(sorted_connections.values()), S=2, curve="convex", direction="increasing")
    return kneedle.knee,kneedle.knee_y

    
def thresh_2deriv(sorted_connections):
    y = list(sorted_connections.values())
    x = np.arange(len(sorted_connections))
    
    dy=np.diff(y,1)
    dx=np.diff(x,1)
    yfirst=dy/dx
    xfirst=0.5*(x[:-1]+x[1:])
    
    dyfirst=np.diff(yfirst,1)
    dxfirst=np.diff(xfirst,1)
    ysecond=dyfirst/dxfirst
    
    xsecond=0.5*(xfirst[:-1]+xfirst[1:])
    
    thresh_x = xsecond[np.argmax(ysecond)]
                       
    difference_array = np.absolute(x-thresh_x)
 
    # find the index of minimum element from the array
    index = difference_array.argmin()
    
    return index, y[index]




