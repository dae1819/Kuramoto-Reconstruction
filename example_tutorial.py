# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:59:37 2023

@author: Dan
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import xgi

import json


from data import theta_timeseries,simulate_simplicial_kuramoto_poisson
from EM_1 import EM_spread
from EM_2 import EM_spread2
from thresholding import thresh_aic



#%% CREATE EXAMPLE SC USING GNM MODEL
n = 10 #8  #nodes
m = 15 #12  #edges
seed = 20189 #854 

# Use seed for reproducibility
G = nx.gnm_random_graph(n, m, seed=seed)


SC = xgi.generators.classic.flag_complex(G, max_order=2)
w = 2*np.random.normal(loc=1, scale=0.3, size=(n,))



#%% ALTERNATIVELY, CREATE TILED SC 

nodes_tile = 7
dim = round(np.sqrt(nodes_tile))


G_tiled = nx.triangular_lattice_graph(dim, dim)
G_tiled = nx.convert_node_labels_to_integers(G_tiled)

#Fill in all possible triangles
SC_tiled = xgi.generators.classic.flag_complex(G_tiled, max_order=2)

#Remove 60 percent of the triangles
removal=0.60
all_tri = list(SC_tiled.edges.filterby("order", 2))
remove = np.random.choice(all_tri,round(len(all_tri)*removal),replace=False)

SC_tiled.remove_simplex_ids_from(remove)





#%% DRAWING

plt.figure(figsize=(10, 4))

# Sequential colormap
cmap = plt.cm.Paired

ax = plt.subplot(1, 2, 1)
pos=xgi.weighted_barycenter_spring_layout(SC)
xgi.drawing.xgi_pylab.draw(SC, pos, ax=ax,edge_fc='green',node_labels=True)
plt.show()



#%% GENERATE DATA FOR % REPEATS
thetas = []
repeats=5
for i in range(repeats):
    print("Generating data for repeat {}".format(i))
    theta_i = np.random.uniform(low=0, high=2*np.pi, size=(n,))
    thetas.append(theta_timeseries(SC, k2 = 4, k3 = 12, w = w, theta = theta_i,suppression=True,timesteps=125000)[0])
    #thetas.append(simulate_simplicial_kuramoto_poisson(SC_json,n_steps=500000,order=0)[0].T)



#%% "2 STEP" INFERENCE SCHEME (SEE MULTITHREADING CODE FOR "FASTER" IMPLEMENTATION)

all_pairs = {}
triangles = {} 

for index, i in enumerate(list(SC.nodes)):
    print("CONNECTIONS FOR NODE {}".format(i))
    print("( Progress ~ {}% )".format(100*i/len(list(SC.nodes))))
    print(">>> (Step 1/2) - Approx neighbours")
    labels = np.delete(np.arange(n),i)
    series_list = [list(EM_spread(theta,i).values()) for theta in thetas]
    aggregate = np.nanmean(series_list,axis=0)
    
    P2 = dict(zip(labels, aggregate))
    
    
    all_pairs.update(dict(zip([frozenset((i,label)) for label in labels], aggregate)))
    
    
    approx_neigh = {k:v for (k,v) in P2.items() if v>=0.3}  
    

    if len(list(approx_neigh.values())) == 0:
        print(">>>>>> Warning: No approx neighbours!")
        approx_neigh = {k:v for (k,v) in P2.items() }
    
    
    P20 = list(approx_neigh.values()) 
    

    keys = np.sort(np.append(list(approx_neigh.keys()),i))
    vals = np.arange(len(keys))
    nodes = dict(zip(keys,vals))
    
    
    theta_filter =  [theta[:,list(nodes.keys())] for theta in thetas]
    print(">>>>>> Approx neighbours: {}".format(list(approx_neigh.keys())))
    print(">>>>>> Probability: {}".format(np.round(P20,2)))
    
    
    print(">>> (Step 2/2) - Three body inference")
    res = np.array([EM_spread2(theta,nodes,i,P20) for theta in theta_filter])
    
    keys_pair = res[:,0][0].keys()
    keys_tri = res[:,1][0].keys()
    vals_pair = np.nanmean(np.array([list(d.values()) for d in res[:,0]]),axis=0)
    vals_tri = np.nanmean(np.array([list(d.values()) for d in res[:,1]]),axis=0)
    
    P2s_final = dict(zip(keys_pair,vals_pair))
    P3s_final = dict(zip(keys_tri,vals_tri))
    
    all_pairs.update(P2s_final)
    triangles.update(P3s_final)   
    

'''
all_pairs and triangles are dictionaries holding all combinations of 
links and triangles
'''
    
    
    
#%% PLOTTING FOR PAIRS 


plt.figure()
plt.xlabel('Links')
plt.ylabel('probability')
plt.title('2-body')

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
            plt.plot(x[i,], P2,'.',color='red')
    else:
            
            if link:
                fp +=1
            plt.plot(x[i], P2,'.',color='blue')
            
            
            

f1 = tp/(tp+0.5*(fp+fn))
print("F1 score: {}".format(f1))
print("tp: {}, fp: {}, fn: {}".format(tp,fp,fn))
plt.plot(np.arange(0, len(x), 1), np.ones(len(x))*0.3, color = 'black')
true_patch = mpatches.Patch(color='red', label='True Link')
false_patch = mpatches.Patch(color='blue', label='False Link')
cut_off_line = plt.Line2D([0], [0], label='Cut-off', color='black',linestyle=':')
plt.legend(handles=[true_patch,false_patch,cut_off_line])

plt.show()



#%% PLOTTING FOR TRIANGLES

plt.figure()
plt.xlabel('index')
plt.ylabel('probability')
plt.title('3-body')

sorted_connections = {k: v for k, v in sorted(triangles.items(), key=lambda item: item[1])}

tp,fp,fn=0,0,0
x = np.arange(0, len(sorted_connections))
for i,P3 in enumerate(list(sorted_connections.values())):
    key = list(sorted_connections.keys())[i]
    if P3 > 0.3:
            link = True 
    else:
            link = False
        
    if frozenset((key)) in SC.edges.members():
            if link:
                tp+=1
            else:
                fn+=1
            plt.plot(x[i,], P3,"^",color='red')
    else:
            
            if link:
                fp +=1
            plt.plot(x[i], P3,"^",color='blue')
          

f1 = tp/(tp+0.5*(fp+fn))
print("F1 score: {}".format(f1))
print("tp: {}, fp: {}, fn: {}".format(tp,fp,fn))
true_patch = mpatches.Patch(color='red', label='True Link')
false_patch = mpatches.Patch(color='blue', label='False Link')
plt.legend(handles=[true_patch,false_patch])

plt.show()





#%% EVALUATING THRESHOLDS (BOUNDARY CLASSIFICATION) - requires libraries kneed and RegScorePy

nums = np.arange(1,len(list(all_pairs.keys())))
aics = [thresh_aic(SC,num,all_pairs) for num in nums]

ps_sorted = np.array(list({k: v for k, v in sorted(all_pairs.items(), key=lambda item: item[1])}.values()))
thresh = [0.5*(np.max(ps_sorted[:-1*num])+ps_sorted[np.argmax(ps_sorted[:-1*num])+1]) for num in nums]



fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

ax1.plot(nums, aics)
ax2.plot(thresh, aics) # Create a dummy plot
ax2.cla()
ax1.set_ylabel("AIC")
ax1.set_xlabel("p")
ax2.set_xlabel("Threshold")
plt.savefig('aic_groups.png', dpi = 720, transparent=True)
plt.show()




