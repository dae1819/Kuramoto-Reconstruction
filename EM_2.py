# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 15:31:34 2023

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

def fn_Q2(S,nodes,i):
    #Will return a list
    theta_i = S[:,nodes[i]]
    S_bar = np.delete(S,nodes[i],axis=1)
    gamma = np.zeros(np.shape(S_bar))
    
    for j,theta_j in enumerate(S_bar.T):
        stack = np.vstack((theta_j,theta_i))
        
        std = np.std(stack,axis=0)
        gamma[:,j] = std
    
    delta_sigma_ij = np.diff(gamma,axis=0)
    
    
    success = np.sum(( (delta_sigma_ij<0) & ( ( (np.diff(S_bar.T,axis=1)!=0) & (np.diff(theta_i)!=0) ).T  ) ).astype('int')  ,axis=0)
    total = np.sum( (( (np.diff(S_bar.T,axis=1)!=0) & (np.diff(theta_i)!=0) ).T).astype('int'),axis=0) 
    
    
    return success/total
 
    
 
def fn_Q3(S,nodes,i):   
    #Will return a dictionary
    
    theta_i = S[:,nodes[i]]
    diff_i = np.diff(theta_i)
    n = np.shape(S)[1]
    dictionary = {}
    
    
    
    nodes_list = np.array(list(nodes.keys()))
    pairs = list(it.combinations(nodes_list[nodes_list!=i],2))
    for pair in pairs:
        theta_j = S[:,nodes[pair[0]]]
        theta_k = S[:,nodes[pair[1]]]
        
        diff_j = np.diff(theta_j)
        diff_k = np.diff(theta_k)
        
        stack = np.vstack((theta_j,theta_k,theta_i))
        std = np.std(stack,axis=0)
        diff = np.diff(std)
        
        sucesses = np.sum(((diff<0)&(diff_i!=0)&(diff_j!=0)&(diff_k!=0)).astype('int')) 
        total = np.sum(((diff_i!=0)&(diff_j!=0)&(diff_k!=0)).astype('int'))          #FLAG
        dictionary[frozenset(pair)] = sucesses/(total)
     
    return dictionary                                                                       
   


def fn_psi(S,nodes,i):
    theta_i = S[:,nodes[i]]
    S_bar = np.delete(S,nodes[i],axis=1)
    psi = (( (np.diff(S_bar.T,axis=1)!=0) & (np.diff(theta_i)!=0) ).T).astype('int')
    
    return psi

def fn_psi2(S,nodes,i):
    theta_i = S[:,nodes[i]]
    diff_i = np.diff(theta_i)
    n = np.shape(S)[1]
    dictionary = {}
    
    nodes_list = np.array(list(nodes.keys()))
    pairs = list(it.combinations(nodes_list[nodes_list!=i],2))
    for pair in pairs:
        theta_j = S[:,nodes[pair[0]]]
        theta_k = S[:,nodes[pair[1]]]
        diff_j = np.diff(theta_j)
        diff_k = np.diff(theta_k)
        dictionary[frozenset(pair)] = ((diff_i!=0)&(diff_j!=0)&(diff_k!=0)).astype('int')
       
    return dictionary



     
def fn_rho_j(i,P2,Q2,psi,eps_i,P3,Q3,psi2):
    #Vectorised in j i.e return matrix where each column is rho_j(t) for a given i
    #psi_align = np.insert(np.delete(psi,i,axis=1),0,0,axis=0)
       
    
    num=(P2*Q2)*psi
    # print(np.shape(num))
    den_1 = np.sum((P2*Q2)*psi,axis=1)
    
    
    
    #den_2 = np.sum([Q3[pair]*np.insert(psi2[pair],0,0,axis=0) for pair in Q3.keys()], axis = 0)
    den_2 = np.sum([ P3[pair]*Q3[pair]*psi2[pair] for pair in Q3.keys()], axis = 0)
    #print(eps_i)
    #print(np.shape(eps_i))
    
    
    
    # den = (den_1 + den_2 + eps_i)[:, np.newaxis]   #+ np.sum(eps_i)
    den = (den_1 + den_2 + eps_i)[:, np.newaxis]   #+ np.sum(eps_i)
    # print("new giovannia")
    # print(np.shape(num/den))
    
    #print((den_1==0)&(den_2==0)&(eps_i==0))
    
    return num/den   


def fn_rho_jk(i,P2,Q2,psi,eps_i,P3,Q3,psi2):
     #Will return a dictionary
    #psi_align = np.insert(np.delete(psi,i,axis=1),0,0,axis=0)
    
    num = {pair: Q3[pair] * psi2[pair] for pair in Q3.keys()}
    #num = {pair: (P3[pair])*Q3[pair] * np.insert(psi2[pair],0,0,axis=0) for pair in Q3.keys()}

    den_1 = np.sum((P2*Q2)*psi,axis=1)
    #den_2 = np.sum([Q3[pair]*np.insert(psi2[pair],0,0,axis=0) for pair in Q3.keys()])
    den_2 = np.sum([(P3[pair])*Q3[pair]*psi2[pair] for pair in Q3.keys()],axis=0)
    den = den_1 + den_2  + eps_i
    final = {pair: num[pair]/(den) for pair in num.keys()}
    
    
    return final


def fn_rho_eps_i(i,P2,Q2,psi,eps_i,P3,Q3,psi2):
    #Will return a list(?)
    #psi_align = np.insert(np.delete(psi,i,axis=1),0,0,axis=0)
    # den_1 = np.sum((P2*Q2)[1:]*psi)
    # den_2 = np.sum([(P3[pair])[1:]*Q3[pair]*psi2[pair] for pair in Q3.keys()])
    # den = den_1 + den_2 + eps_i
    
    
    den_1 = np.sum((P2*Q2)*psi,axis=1)
    den_2 = np.sum([(P3[pair])*Q3[pair]*psi2[pair] for pair in Q3.keys()],axis=0)
    den = (den_1 + den_2 + eps_i) #
    

    return eps_i/den
    


def fn_P2(i,S,nodes,rho_j,Q2,psi):
    
    theta_i = S[:,nodes[i]]
    S_bar = np.delete(S,nodes[i],axis=1)
    gamma = np.zeros(np.shape(S_bar))
    
    for j,theta_j in enumerate(S_bar.T):
        stack = np.vstack((theta_j,theta_i))
        
        std = np.std(stack,axis=0)
        gamma[:,j] = std
        
    delta_sigma_ij = np.diff(gamma,axis=0)
    
    den=np.sum( ( ( (  (np.diff(S_bar.T,axis=1)!=0) & (np.diff(theta_i)!=0)  ).T ).astype('int'))*Q2,axis=0)

    num = np.sum(rho_j*(delta_sigma_ij<0).astype('int'), axis =0)   #tm+1
    #den= np.sum(P2*psi_align*conj[:, np.newaxis], axis = 0)
    #print(psi_align)
    return num/den

def fn_P3(i,S,nodes,rho_jk,psi,Q3,psi2):
    P3 = {}
    #psi_long = np.insert(psi[:,i],0,0,axis=0)
    #conj = np.where((psi_long==0)|(psi_long==1), psi_long^1, psi_long)
    
    
    
    theta_i = S[:,nodes[i]]
    diff_i = np.diff(theta_i)
    n = np.shape(S)[1]
    delta_sigma_3 = {}
    cond = {}
    
    nodes_list = np.array(list(nodes.keys()))
    pairs = list(it.combinations(nodes_list[nodes_list!=i],2))
    for pair in pairs:
        theta_j = S[:,nodes[pair[0]]]
        theta_k = S[:,nodes[pair[1]]]
        diff_j = np.diff(theta_j)
        diff_k = np.diff(theta_k)
        
        stack = np.vstack((theta_i,theta_j,theta_k))
        std = np.std(stack,axis=0)
        delta_sigma_3[frozenset(pair)] = np.diff(std)
        cond[frozenset(pair)] = (diff_i!=0)&(diff_j!=0)&(diff_k!=0)
        
    for pair in psi2.keys():
        #num = np.sum(rho_jk[pair]*np.append(psi[:,i],0)*conj)
        #den = np.sum(Q3[pair]*np.insert(psi2[pair],0,0)*conj)
        #print(delta_sigma_3)
        num = np.sum(rho_jk[pair]*(delta_sigma_3[pair]<0).astype('int'), axis =0)
        #den=np.sum( ( ( (  (np.diff(S_bar.T,axis=1)!=0) & (np.diff(theta_i)!=0)  ).T ).astype('int'))*Q3,axis=0)
        den = np.sum(cond[pair]*Q3[pair],axis=0)
        P3[pair] = num/den
    

    return P3

def fn_eps(i,S,nodes,rho_eps, psi):
    theta_i = S[:,nodes[i]]
    den=np.sum(( (np.diff(theta_i)!=0).astype('int')),axis=0)    
    num = np.sum(rho_eps *(np.diff(theta_i)!=0).astype('int'))   #tm+1

    return (num/den)#*np.ones(len(theta_i)-1)[:, np.newaxis] 

  
def EM_spread2(S,nodes,i,P20):
    #n = np.shape(S)[1]
    m,n = S.shape

    #Notation: _ is ->
    # P2 : Pj->i,   P3 : Pjk->i,   Q2 : Pij,   Q3 : Pijk 

    Q2 = fn_Q2(S,nodes,i)
    Q3 = fn_Q3(S,nodes,i)
    
    psi = fn_psi(S,nodes,i)
    psi2 = fn_psi2(S,nodes,i)
    
    P2,eps_i = P20,np.ones(m-1)
    
    P3 = {}
    z = {}
    
    nodes_list = np.array(list(nodes.keys()))
    pairs = list(it.combinations(nodes_list[nodes_list!=i],2))
    
 
    for pair in pairs:
        P3[frozenset(pair)] = 1 #np.ones(m-1)
        z[frozenset(pair)] = 0
    
    
       
    
    v = np.zeros(n-1) #just dummy variables to specify convergence
    k = 0

    
    
    while (np.abs(v-P2)>0.001).all():
        
        
        v = P2
        z = P3
        
        rho_j = fn_rho_j(i,P2,Q2,psi,eps_i,P3,Q3,psi2)
        rho_jk = fn_rho_jk(i,P2,Q2,psi,eps_i,P3,Q3,psi2)
        rho_eps = fn_rho_eps_i(i,P2,Q2,psi,eps_i,P3,Q3,psi2)
        
        
        P2 = fn_P2(i,S,nodes,rho_j,P2,psi) 
        P3 = fn_P3(i,S,nodes,rho_jk,psi,Q3,psi2) 
        
        eps_i = fn_eps(i,S,nodes,rho_eps, psi)
                 
        k = k+1
        
        if k>1000:
            print(">>>>>> Non-conv step")
            P2 += np.nan
            for key in P3.keys():
                P3[key]=np.nan
            break
        
        
        
        
       

    P2_labels =  nodes_list[nodes_list!=i]
    P2_labelled = {}
    
    for indx,label in enumerate(P2_labels):
        P2_labelled[frozenset((i,label))]=P2[indx]
    
        
    
    
    P3_labelled = {}
    
   
    
    for pair in P3.keys():
        label = np.concatenate(([i],list(pair)))
        P3_labelled[frozenset(label)] = P3[pair]
        
        
    
    
    
    
    return [P2_labelled,P3_labelled]
                
    # for i in range(j):
    #     rhoj = fn_rhoj(j,epsi,P2,P2_,P3,P3_, t)
    #     for j in range(k):
    #         rhojk = fn_rhojk(j,epsi,P2,P2_,P3,P3_, t)
    #         rhoeps = fn_rhoepsi(j,k,epsi,P2,P2_,P3,P3_, t)
    
    # P2_ = fn_P2_(j,P2,P2_,rhoj, t)
    # P3_ = fn_P3_(j,P3,P3_,rho_jk, t)
    # epsi
    
    # pass


    # while np.sum(np.abs(p-p1))>0.0001 or np.sum(np.abs(q-q1))>0.0001:


