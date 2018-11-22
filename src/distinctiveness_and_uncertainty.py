#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:46:50 2018

@author: ziyueli
"""

import numpy as np
#from numpy.linalg import norm
import scipy.spatial.distance as dist
import scipy.stats as stats

import time

check = False
start_time = time.time()

#input x_A, x_B, c_A, c_B, alpha
#def 2d_matrix_initial(n_row, n_column):
#    return np.zeros(())
#---------------------------------input data-----------------------------------
if check:
    n_classes = 1000
    batch_size = 100
    N_A = 9216
    N_B = 4096
    x_A = np.random.randn(batch_size, N_A)
    x_B = np.random.randn(batch_size, N_B)
    c_A = np.random.randn(n_classes, N_A)
    c_B = np.random.randn(n_classes, N_B)
    alpha = np.ones((batch_size, n_classes))/n_classes
    print("n_classes =  {}\
        batch_size = {}\
        N_A ={}\
        N_B ={}".format(n_classes, batch_size,  N_A, N_B))
#----------------- calculate Sx_A, Sx_B, Sx_A_to_B -------------------------------
def Sx_generator(x,c):
    #x = x[:, np.newaxis, :]
    #c = c[np.newaxis, :, :]
    #i = x - c
    #n,m,o = i.shape
    d = dist.cdist(x, c, 'euclidean')#i.reshape(n*m,o),np.zeros((m,o))).reshape(n,m,m).diagonal(axis1=2).reshape(n,m)
    return d
    # i[batch_size, 1000, N_A], d[batch_size, 1000]

##quick example for dist.cdist function    
#i = np.array([[[1, 1],
#               [1, 1],
#               [0, 0]],
#              [[2, 2],
#               [2, 2],
#               [2, 2]],
#              [[2, 1],
#               [2, 1],
#               [2, 1]],
#              [[3, 3],
#               [3, 3],
#               [1, 1]]])
#n,m,o = i.shape
#dist.cdist(i.reshape(n*m,o),np.zeros((m,o))).reshape(n,m,m).diagonal(axis1=2).reshape(n,m)


def distinct(alpha, x_A, x_B, c_A, c_B, Sc_A2B=None):
    # ----------------------- calculate  S_hat_x_A_to_B -------------------------------
    if Sc_A2B is None:
        ScA = Sx_generator(c_A, c_A)
        ScB = Sx_generator(c_B, c_B)
        Sc_A2B = ScA - ScB
    S_hat_x_A2B = alpha.dot(Sc_A2B.T)

    # ----------------------- calculate  S_x_A_to_B -------------------------------
    Sx_A = Sx_generator(x_A, c_A)
    Sx_B = Sx_generator(x_B, c_B)
    Sx_A2B = Sx_A - Sx_B

    # ----------------------- calculate  distinctness -------------------------------
    return np.array([(1 - stats.kendalltau(x, x_approx)[0])/2
                     for x, x_approx in zip(Sx_A2B, S_hat_x_A2B)])

if False:
    start_time = time.time()
    distinct(alpha, x_A, x_B, c_A, c_B)
    print("Time without Sc_A2B: ", time.time() - start_time, flush=True)

    ScA = Sx_generator(c_A, c_A)
    ScB = Sx_generator(c_B, c_B)
    Sc_A2B = ScA - ScB

    start_time = time.time()
    distinct(alpha, x_A, x_B, c_A, c_B, Sc_A2B=Sc_A2B)
    print("Time with Sc_A2B: ", time.time() - start_time, flush=True)


#----------------------- calculate  uncertainty -------------------------------
if check:
    K_new = 2
    p = np.ones((batch_size, K_new))/batch_size

def uncertainty(p):
    return(np.sum(p * (1 - p), axis=-1))

if check:
    start_time = time.time()
    uncertainty(p)
    print("Uncertainty check: ", time.time() - start_time)

# ----------------------- calculate  score -------------------------------
def score(lamb, t, p, alpha, x_A, x_B, c_A, c_B, Sc_A2B=None):
    if (1-lamb*t) <= 0:
        return uncertainty(p)
    else:
        return (1-lamb*t)* distinct(alpha, x_A, x_B, c_A, c_B, Sc_A2B=Sc_A2B) + lamb*t * uncertainty(p)

if check:
    lamb = 0.9
    t = 100

    start  = time.time()
    ScA = Sx_generator(c_A, c_A)
    ScB = Sx_generator(c_B, c_B)
    Sc_A2B = ScA - ScB
    print("Time on Sc_A2B construction: {:}".format(time.time() - start_time))

    start_time = time.time()
    sc = score(lamb, t, p, alpha, x_A, x_B, c_A, c_B, Sc_A2B=Sc_A2B)
    print("Score check: {:}, value: {:}".format(time.time() - start_time, sc))




