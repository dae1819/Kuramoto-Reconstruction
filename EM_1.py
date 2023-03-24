# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 15:30:28 2023

@author: Dan
"""



import numpy as np
import xgi
import random
import networkx as nx
import scipy.special as special
import itertools as it
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from collections import Counter

import pandas as pd


def fn_Q2(S,i): 
    #Will return a list
    theta_i = S[:,i]
    S_bar = np.delete(S,i,axis=1)
    gamma = np.zeros(np.shape(S_bar))
    
    for j,theta_j in enumerate(S_bar.T):
        stack = np.vstack((theta_j,theta_i))
        
        std = np.std(stack,axis=0)
        gamma[:,j] = std
    
    delta_sigma_ij = np.diff(gamma,axis=0)
    #delta_sigma_ij = np.sin(np.diff(gamma,axis=0))
    
    success = np.sum(( (delta_sigma_ij<0) & ( ( (np.diff(S_bar.T,axis=1)!=0) & (np.diff(theta_i)!=0) ).T  ) ).astype('int')  ,axis=0)
    total = np.sum( (( (np.diff(S_bar.T,axis=1)!=0) & (np.diff(theta_i)!=0) ).T).astype('int'),axis=0) 
    
    
    return success/total
 
    

def fn_psi(S,i):
    theta_i = S[:,i]
    S_bar = np.delete(S,i,axis=1)
    psi = (( (np.diff(S_bar.T,axis=1)!=0) & (np.diff(theta_i)!=0) ).T).astype('int')
    
    return psi

 
def fn_rho_j(i,P2,Q2,psi,eps_i):
    num=(P2*Q2)*psi
    den_1 = np.sum((P2*Q2)*psi,axis=1)
    den = (den_1 + eps_i)[:, np.newaxis]   
   
    return num/den   


def fn_rho_eps_i(i,P2,Q2,psi,eps_i):
    den_1 = np.sum((P2*Q2)*psi,axis=1)
    den = (den_1 + eps_i) 

    return eps_i/den
    


def fn_P2(i,S,rho_j,Q2,psi):
    
    theta_i = S[:,i]
    S_bar = np.delete(S,i,axis=1)
    gamma = np.zeros(np.shape(S_bar))
    
    for j,theta_j in enumerate(S_bar.T):
        stack = np.vstack((theta_j,theta_i))
        
        std = np.std(stack,axis=0)
        gamma[:,j] = std
        
    delta_sigma_ij = np.diff(gamma,axis=0)
    #delta_sigma_ij = np.sin(np.diff(gamma,axis=0))
    
    
    den=np.sum( ( ( (  (np.diff(S_bar.T,axis=1)!=0) & (np.diff(theta_i)!=0)  ).T ).astype('int'))*Q2,axis=0)

    num = np.sum(rho_j*(delta_sigma_ij<0).astype('int'), axis =0)   #tm+1
    
    return num/den


def fn_eps(i,S,rho_eps, psi):
   
    theta_i = S[:,i]
    # S_bar = np.delete(S,i,axis=1)
    # gamma = np.zeros(np.shape(S_bar))
    
    # for j,theta_j in enumerate(S_bar.T):
    #     stack = np.vstack((theta_j,theta_i))
        
    #     std = np.std(stack,axis=0)
    #     gamma[:,j] = std
        
    # delta_sigma_ij = np.diff(gamma,axis=0) 
    # #delta_sigma_ij = np.sin(np.diff(gamma,axis=0))
    
    
    den=np.sum(( (np.diff(theta_i)!=0).astype('int')),axis=0)    
    
    num = np.sum(rho_eps *(np.diff(theta_i)!=0).astype('int'))   #tm+1

    
    
    
    
    return (num/den) 


def EM_spread(S,i):
    #n = np.shape(S)[1]
    m,n = np.shape(S)
    #Notation: _ is ->
    # P2 : Pj->i,   P3 : Pjk->i,   Q2 : Pij,   Q3 : Pijk 

    Q2 = fn_Q2(S,i)

    
    psi = fn_psi(S,i)
   
    
    P2,eps_i = np.ones(n-1)*0.5,np.ones(m-1)
    

        
    v = np.zeros(n-1) #just dummy variables to specify convergence
   
    
    
    
    P2s = np.zeros(n-1)
    for j in range(1,3):
        P2s = np.vstack((P2s,j*np.ones(n-1)))
        
        
    k=0
    punched = False
    
    while (np.abs(v-P2)>0.001).all():
    #while (np.abs(np.std(P2s[-3:,:],axis=0))>0.003).all():   
    #for k in range(0,lim):
    #     print("k = {}/{}, {}%".format(k,lim,100*k/lim))
        
        #print(k)
        v = P2
        
        rho_j = fn_rho_j(i,P2,Q2,psi,eps_i)
        
        rho_eps = fn_rho_eps_i(i,P2,Q2,psi,eps_i)
        P2 = fn_P2(i,S,rho_j,P2,psi) 
        eps_i = fn_eps(i,S,rho_eps, psi)
        
        
        # plt.plot(k,P2[5],"x",color="red")
        # plt.plot(k,P2[6],"x",color="green")
        # plt.plot(k,P2[7],"x",color="blue")
        
        # if k == lim-1:
        #     P2[np.abs(P2-v)>0.1]=np.nan    
        
        # if (np.abs(v-P2)<0.0005).all():
        #     break
        
        #No real difference for inclusion of below
        # if k > round(200) and punched==False:
        #     punched=True
        #     #non_conv = np.where((np.abs(np.std(P2s[-3:,:],axis=0))>0.003))
        #     non_conv = np.where(np.abs(v-P2)>0.00075)[0]
        #     print("Number of non-converging nodes: {}".format(len(non_conv)))
        #     #diff = np.abs((P2-v)[non_conv])/4
        #     diff = 0.25
        #     P2[non_conv] = P2[non_conv] - np.sign((P2-v)[non_conv])*P2[non_conv]*diff
        
        
        
        
        #print(P2)
     
        
        P2s = np.vstack((P2s,P2))
        
        k+=1
         
        if k>1000:
            P2 += np.nan
            break
        
    
    P2_labels = np.delete(np.arange(n),i)
    P2_labelled = {}
    
    for indx,label in enumerate(P2_labels):
        P2_labelled[frozenset((i,label))]=P2[indx]
    
    return P2_labelled
                
     