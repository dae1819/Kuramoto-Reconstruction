# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:06:42 2023

@author: Dan
"""


"""
This example details the use of multithreading, in particular to make 
a heatmap of the parameter space performance
"""


import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import xgi
import json
import seaborn as sns

from data import theta_timeseries,simulate_simplicial_kuramoto_poisson
from EM_1 import EM_spread
from EM_2 import EM_spread2


def step(coord,SC,indx):
    n = len(SC.nodes)
    thetas = []
    repeats=5
    for i in range(repeats):
        print("Generating data for repeat {}".format(i))
        theta_i = np.random.uniform(low=0, high=2*np.pi, size=(n,))
        w = np.ones(n)
        thetas.append(theta_timeseries(SC, k2 = coord[0], k3 = coord[1], w = w, theta = theta_i,suppression=True,timesteps=200000)[0])
        
    all_pairs = {}
    triangles = {} 
    for index, i in enumerate(list(SC.nodes)):
        # Approx neighbours
        labels = np.delete(np.arange(n),i)
        series_list = [list(EM_spread(theta,i).values()) for theta in thetas]
        aggregate = np.nanmean(series_list,axis=0)
        
        P2 = dict(zip(labels, aggregate))
        
        all_pairs.update(dict(zip([frozenset((i,label)) for label in labels], aggregate)))
        
        approx_neigh = {k:v for (k,v) in P2.items() if v>=0.3}
        P20 = list(approx_neigh.values())
        
        keys = np.sort(np.append(list(approx_neigh.keys()),i))
        vals = np.arange(len(keys))
        nodes = dict(zip(keys,vals))
        
        theta_filter =  [theta[:,list(nodes.keys())] for theta in thetas]
        
        # Triangles
        res = np.array([EM_spread2(theta,nodes,i,P20) for theta in theta_filter])
        
        keys_pair = res[:,0][0].keys()
        keys_tri = res[:,1][0].keys()
        vals_pair = np.nanmean(np.array([list(d.values()) for d in res[:,0]]),axis=0)
        vals_tri = np.nanmean(np.array([list(d.values()) for d in res[:,1]]),axis=0)
        
        P2s_final = dict(zip(keys_pair,vals_pair))
        P3s_final = dict(zip(keys_tri,vals_tri))
        
        all_pairs.update(P2s_final)
        triangles.update(P3s_final)
        
    
    save_pairs = {str(list(k)):v for (k,v) in all_pairs.items()}
    save_tri = {str(list(k)):v for (k,v) in triangles.items()}
    
    #Save the pair and triangle probabilities
    with open("results/SC_pairs_{}.txt".format(indx), "w") as convert_file:
        convert_file.write(json.dumps(save_pairs))
    with open("results/SC_tri_{}.txt".format(indx), "w") as convert_file:
        convert_file.write(json.dumps(save_tri))
    
    sorted_connections = {k: v for k, v in sorted(all_pairs.items(), key=lambda item: item[1])}

    tp,fp,fn=0,0,0
    x = np.arange(0, len(sorted_connections))
    for i,P2 in enumerate(list(sorted_connections.values())):
        key = list(sorted_connections.keys())[i]
        if P2 > 0.3:
                link = True 
        else:
                link = False
            
        if frozenset((key)) in SC.edges.filterby("size", 2).members():
                if link:
                    tp+=1
                else:
                    fn+=1      
        else:
                if link:
                    fp +=1
        
                    
    f1 = tp/(tp+0.5*(fp+fn))
    return f1





if __name__ == "__main__":
    #Define processiong pool. Currently uses all available cores
    pool = multiprocessing.Pool()
    
    #DEFINE SC HERE
    n = 10 #8  #nodes
    m = 15 #12  #edges
    seed = 20189 #854
    G = nx.gnm_random_graph(n, m, seed=seed)
    SC = xgi.generators.classic.flag_complex(G, max_order=2)
    
    #DEFINE COORDINATE SYSTEM
    k2 = np.linspace(0.5,9,8)
    k3 = np.linspace(0,23,8)
    kk2,kk3 = np.meshgrid(k2,k3)
    coords = np.vstack([kk2.ravel(), kk3.ravel()]).T
    
    #Multithread execution
    processes = [pool.apply_async(step, args=(coord,SC,indx)) for indx,coord in enumerate(coords)]
    f1s = np.array([p.get() for p in processes])
    np.savetxt("results/f1s.txt",f1s)

    #Make the heatmap
    M = f1s.reshape((8,8))
    M = np.flip(M,axis=0)
    
    h = sns.heatmap(M,xticklabels=k2, yticklabels=np.flip(k3))
    h.set(xlabel=r'$K_2$', ylabel=r'$K_3$')
    



