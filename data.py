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



def Extract_Degree(SC):
    #Calculate k_q = mean number q-simplex degree
    #Mean number of times node i appeared in link of rank q
    n = len(SC.nodes)

    edges_1 = list(SC.edges.filterby("order", 1))
    a = np.array([SC.edges.members(x) for x in edges_1])
    el= []
    for sets in a:
        for element in sets:
            el.append(element)
    #counts= Counter(a)
    counts = np.array([el.count(x) for x in range(n)])
    
    q1 = np.mean(counts)
    
    edges_2 = list(SC.edges.filterby("order", 2))
    a2 = np.array([SC.edges.members(x) for x in edges_2])
    el2= []
    for sets in a2:
        for element in sets:
            el2.append(element)
    #counts= Counter(a)
    counts2 = np.array([el2.count(x) for x in range(n)])
    q2 = np.mean(counts2)
    
    return q1, q2


def theta_timeseries(SC, k2, k3, w, theta, timesteps=20000, dt=0.002,suppression=True):
    q1 = Extract_Degree(SC)[0]
    q2 = Extract_Degree(SC)[1]
   
    
    thetas = [theta]
    
    
    SC_int = xgi.convert_labels_to_integers(SC, "label")

    links = SC_int.edges.filterby("size", 2).members()
    
    
    
    triangles = SC_int.edges.filterby("size", 3).members()
    n_triangles = len(SC_int.edges.filterby("size", 3).members())
    
    
    
    n = SC_int.num_nodes
    
    
    R_time = np.zeros(timesteps)
    peturb_time = np.zeros(timesteps)

    ints = np.zeros((timesteps,n))    


    num_int_2 = np.random.poisson(0.3,timesteps)   #b1 = 0.8/5 ~ 0.16
    num_int_3 = np.random.poisson(0.6,timesteps)   #b2 = 4/1.5 ~ 2.7
    # num_int_2[num_int_2>n_links]=n_links
    # num_int_3[num_int_3>n_triangles]=n_triangles
     
    
    for t in range(timesteps):
        
        
        r1 = np.zeros(n, dtype=complex)
        r2 = np.zeros(n, dtype=complex)

        int_links = np.random.choice(links,num_int_2[t])#,replace=False) 
        for i, j in int_links:
            
            ints[t,i] += 1
            ints[t,j] += 1 
            
            r1[i] += np.exp(1j * theta[j])
            r1[j] += np.exp(1j * theta[i])
            
            
            #state(neig(   rand(1,length(neig))<beta1     ))=1;
        if n_triangles >0:
            int_triangles = np.random.choice(triangles,num_int_3[t])#,replace=False)
            for i, j, k in int_triangles:
                
                ints[t,i] += 1
                ints[t,j] += 1 
                ints[t,k] += 1 
                
                r2[i] += np.exp(2j * theta[j] - 1j * theta[k]) + np.exp(
                        2j * theta[k] - 1j * theta[j]
                )
                
                r2[j] += np.exp(2j * theta[i] - 1j * theta[k]) + np.exp(
                        2j * theta[k] - 1j * theta[i]
                )
                
                
                r2[k] += np.exp(2j * theta[i] - 1j * theta[j]) + np.exp(
                        2j * theta[j] - 1j * theta[i]
                )

    
            d_theta = (
                #w +
                 (k2/q1) * np.multiply(r1, np.exp(-1j * theta)).imag
                + (k3/(2*q2)) * np.multiply(r2, np.exp(-1j * theta)).imag
            )
        else:
            d_theta = (
                #w +
                 (k2/q1) * np.multiply(r1, np.exp(-1j * theta)).imag)
        
        theta_new = theta + d_theta * dt
        
        z = np.mean(np.exp(1j * theta_new))
        R_time[t] = np.abs(z)
        
        
        if suppression:
            if np.abs(z)>=0.9:
                theta = np.random.uniform(low=0, high=2*np.pi, size=(n,)) 
                peturb_time[t] = 1
                print('SUP')
            else:
                theta = theta_new
                
        else:
            theta = theta_new
        
        thetas.append(theta)
        
    print(k2/q1)
    print(k3/(2*q2))    
    return np.array(thetas),R_time,peturb_time,ints



def simulate_simplicial_kuramoto(
    S,
    orientations=None,
    order=1,
    sigma=1,
    T=10,
    n_steps=10000,
    index=False,
):
    """
    This function simulates the simplicial Kuramoto model's dynamics on an oriented simplicial complex
    using explicit Euler numerical integration scheme.
    Parameters
    ----------
    S: simplicial complex object
        The simplicial complex on which you
        run the simplicial Kuramoto model
    orientations: dict, Default : None
        Dictionary mapping non-singleton simplices IDs to their boolean orientation
    order: integer
        The order of the oscillating simplices
    omega: numpy.ndarray
        The simplicial oscillators' natural frequencies, has dimension
        (n_simplices of given order, 1)
    sigma: positive real value
        The coupling strength
    theta0: numpy.ndarray
        The initial phase distribution, has dimension
        (n_simplices of given order, 1)
    T: positive real value
        The final simulation time.
    n_steps: integer greater than 1
        The number of integration timesteps for
        the explicit Euler method.
    index: bool, default: False
        Specifies whether to output dictionaries mapping the node and edge IDs to indices
    Returns
    -------
    theta: numpy.ndarray
        Timeseries of the simplicial oscillators' phases, has dimension
        (n_simplices of given order, n_steps)
    theta_minus: numpy array of floats
        Timeseries of the projection of the phases onto lower order simplices,
        has dimension (n_simplices of given order - 1, n_steps)
    theta_plus: numpy array of floats
        Timeseries of the projection of the phases onto higher order simplices,
        has dimension (n_simplices of given order + 1, n_steps)
    om1_dict: dict
        The dictionary mapping indices to (order-1)-simplices IDs, if index is True
    o_dict: dict
        The dictionary mapping indices to (order)-simplices IDs, if index is True
    op1_dict: dict
        The dictionary mapping indices to (order+1)-simplices IDs, if index is True
    References
    ----------
    "Explosive Higher-Order Kuramoto Dynamics on Simplicial Complexes"
    by Ana P. Millán, Joaquín J. Torres, and Ginestra Bianconi
    https://doi.org/10.1103/PhysRevLett.124.218301
    """

    # Notation:
    # B_o - boundary matrix acting on (order)-simplices
    # D_o - adjoint boundary matrix acting on (order)-simplices
    # om1 = order - 1
    # op1 = order + 1

    if not isinstance(S, xgi.SimplicialComplex):
        raise XGIError(
            "The simplicial Kuramoto model can be simulated only on a SimplicialComplex object"
        )

    if index:
        B_o, om1_dict, o_dict = xgi.matrix.boundary_matrix(S, order, orientations, True)
    else:
        B_o = xgi.matrix.boundary_matrix(S, order, orientations, False)
    D_om1 = np.transpose(B_o)

    if index:
        B_op1, __, op1_dict = xgi.matrix.boundary_matrix(
            S, order + 1, orientations, True
        )
    else:
        B_op1 = xgi.matrix.boundary_matrix(S, order + 1, orientations, False)
    D_o = np.transpose(B_op1)

    # Compute the number of oscillating simplices
    n_o = np.shape(B_o)[1]
    
    #Got rid of omega for now and added theta_0 (intial phase on edges inside the function)
    
    theta0=np.random.uniform(low=0, high=2*np.pi, size=(1,n_o))
    
    
    dt = T / n_steps
    theta = np.zeros((n_o, n_steps))
    theta[:, 0] = theta0
    
    
    for t in range(1, n_steps):
        theta[:, t] = theta[:, t-1] + dt * (
            - sigma * D_om1 @ np.sin(B_o @ theta[:, t-1])
            - sigma * B_op1 @ np.sin(D_o @ theta[:, t-1])
        )
     
    
    theta_minus = B_o @ theta
    theta_plus = D_o @ theta
    if index:
        return theta, theta_minus, theta_plus, om1_dict, o_dict, op1_dict
    else:
        return theta, theta_minus, theta_plus

def simulate_simplicial_kuramoto_poisson(
    S,
    orientations=None,
    order=1,
    sigma=1,
    T=10,
    n_steps=10000,
    index=False,
):
    """
    This function simulates the simplicial Kuramoto model's dynamics on an oriented simplicial complex
    using explicit Euler numerical integration scheme.
    Parameters
    ----------
    S: simplicial complex object
        The simplicial complex on which you
        run the simplicial Kuramoto model
    orientations: dict, Default : None
        Dictionary mapping non-singleton simplices IDs to their boolean orientation
    order: integer
        The order of the oscillating simplices
    omega: numpy.ndarray
        The simplicial oscillators' natural frequencies, has dimension
        (n_simplices of given order, 1)
    sigma: positive real value
        The coupling strength
    theta0: numpy.ndarray
        The initial phase distribution, has dimension
        (n_simplices of given order, 1)
    T: positive real value
        The final simulation time.
    n_steps: integer greater than 1
        The number of integration timesteps for
        the explicit Euler method.
    index: bool, default: False
        Specifies whether to output dictionaries mapping the node and edge IDs to indices
    Returns
    -------
    theta: numpy.ndarray
        Timeseries of the simplicial oscillators' phases, has dimension
        (n_simplices of given order, n_steps)
    theta_minus: numpy array of floats
        Timeseries of the projection of the phases onto lower order simplices,
        has dimension (n_simplices of given order - 1, n_steps)
    theta_plus: numpy array of floats
        Timeseries of the projection of the phases onto higher order simplices,
        has dimension (n_simplices of given order + 1, n_steps)
    om1_dict: dict
        The dictionary mapping indices to (order-1)-simplices IDs, if index is True
    o_dict: dict
        The dictionary mapping indices to (order)-simplices IDs, if index is True
    op1_dict: dict
        The dictionary mapping indices to (order+1)-simplices IDs, if index is True
    References
    ----------
    "Explosive Higher-Order Kuramoto Dynamics on Simplicial Complexes"
    by Ana P. Millán, Joaquín J. Torres, and Ginestra Bianconi
    https://doi.org/10.1103/PhysRevLett.124.218301
    """

    # Notation:
    # B_o - boundary matrix acting on (order)-simplices
    # D_o - adjoint boundary matrix acting on (order)-simplices
    # om1 = order - 1
    # op1 = order + 1



    if index:
        B_o, om1_dict, o_dict = xgi.matrix.boundary_matrix(S, order, orientations, True)
    else:
        B_o = xgi.matrix.boundary_matrix(S, order, orientations, False)
    D_om1 = np.transpose(B_o)

    if index:
        B_op1, __, op1_dict = xgi.matrix.boundary_matrix(
            S, order + 1, orientations, True
        )
    else:
        B_op1 = xgi.matrix.boundary_matrix(S, order + 1, orientations, False)
    D_o = np.transpose(B_op1)

    # Compute the number of oscillating simplices
    n_o = np.shape(B_o)[1]
    n_o2 = np.shape(B_op1)[1]
    #Got rid of omega for now and added theta_0 (intial phase on edges inside the function)
    
    theta0=np.random.uniform(low=0, high=2*np.pi, size=(1,n_o))
    
    
    num_int_2 = np.random.poisson(2,n_steps)  
    probs = np.zeros((n_o, n_steps))
    ints = np.arange(n_o)
    for i in range((n_steps)):
        ints_id= np.random.choice(ints,num_int_2[i])#,replace=False) 
        probs[ints_id, i] +=1
        
    
    # num_int_3 = np.random.poisson(0.6,n_steps)  
    # probs2 = np.zeros((n_o2, n_steps))
    # ints = np.arange(n_o2)
    # for i in range((n_steps)):
    #     ints_id= np.random.choice(ints,num_int_3[i])#,replace=False) 
    #     probs2[ints_id, i] +=1

    dt = T / n_steps
    theta = np.zeros((n_o, n_steps))
    theta[:, 0] = theta0
    for t in range(1, n_steps):
        theta[:, t] = theta[:, t-1] + dt* probs[:, t] *(
            - sigma * D_om1 @ np.sin(B_o @ theta[:, t-1])
            - sigma * B_op1 @ np.sin(D_o @ theta[:, t-1])
        )
    
    theta_minus = B_o @ theta
    theta_plus = D_o @ theta
    if index:
        return theta, theta_minus, theta_plus, om1_dict, o_dict, op1_dict
    else:
        return theta, theta_minus, theta_plus