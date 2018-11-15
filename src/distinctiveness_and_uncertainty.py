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

check = True
start_time = time.time()

#input x_A, x_B, c_A, c_B, alpha
#def 2d_matrix_initial(n_row, n_column):
#    return np.zeros(())
#---------------------------------input data-----------------------------------
if check:
    n_classes = 1000
    batch_size = 10
    N_A = 9216
    N_B = 4096
    x_A = np.ones((batch_size, N_A))
    x_B = np.ones((batch_size, N_B))
    c_A = np.ones((n_classes, N_A))
    c_B = np.zeros((n_classes, N_B))
    alpha = np.zeros((batch_size, n_classes))
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


def distinctness(alpha, x_A, x_B, c_A, c_B, Sc_A2B=None):
    # ----------------------- calculate  S_hat_x_A_to_B -------------------------------
    if Sc_A2B is None:
        ScA = Sx_generator(c_A, c_A)
        ScB = Sx_generator(c_B, c_B)
        Sc_A2B = ScA - ScB
    S_hat_x_A2B = np.sum(alpha[:, np.newaxis, :] * Sc_A2B[np.newaxis, :, :], axis=1)
    # ----------------------- calculate  S_x_A_to_B -------------------------------
    Sx_A = Sx_generator(x_A, c_A)
    Sx_B = Sx_generator(x_B, c_B)
    Sx_A2B = Sx_A - Sx_B
    # ----------------------- calculate  distinctness -------------------------------
    return (1 - stats.kendalltau(Sx_A2B, S_hat_x_A2B)[0])/2

if check:
    distinctness(alpha, x_A, x_B, c_A, c_B)
    print("Time without Sc_A2B: ", time.time() - start_time)

    start_time = time.time()
    ScA = Sx_generator(c_A, c_A)
    ScB = Sx_generator(c_B, c_B)
    Sc_A2B = ScA - ScB
    distinctness(alpha, x_A, x_B, c_A, c_B, Sc_A2B=Sc_A2B)
    print("Time with Sc_A2B: ", time.time() - start_time)


#----------------------- calculate  uncertainty -------------------------------
if check:
    K_new = 2
    p = np.zeros((batch_size, K_new))

def uncertainty(p):
    return(-1 * np.sum(p * (1 - p), axis=-1))

if check:
    start_time = time.time()
    uncertainty(p)
    print("Uncertainty check: ", time.time() - start_time)

def score(lamb, t, p, alpha, x_A, x_B, c_A, c_B, Sc_A2B=None):
    return lamb**t * distinctness(alpha, x_A, x_B, c_A, c_B, Sc_A2B=Sc_A2B) + (1-lamb**t) * uncertainty(p)


