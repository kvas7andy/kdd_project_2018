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

#input x_A, x_B, c_A, c_B, alpha
#def 2d_matrix_initial(n_row, n_column):
#    return np.zeros(())
#---------------------------------input data-----------------------------------
batch_size = 10
N_A = 100
N_B = 100
x_A = np.zeros((batch_size, N_A))
x_B = np.zeros((batch_size, N_A))
c_A = np.zeros((1000, N_A))
c_B = np.zeros((1000, N_B))
alpha = np.zeros(batch_size, 1000)
#----------------- calculate Sx_A, Sx_B, Sx_A_to_B -------------------------------
def Sx_generator(x,c):
    x = x[:, np.newaxis, :]
    c = c[np.newaxis, :, :]
    i = x - c
    n,m,o = i.shape
    d = dist.cdist(i.reshape(n*m,o),np.zeros((m,o))).reshape(n,m,m).diagonal(axis1=2).reshape(n,m)
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

Sx_A = Sx_generator(x_A, c_A)
Sx_B = Sx_generator(x_B, c_B)
Sx_A2B = Sx_A - Sx_B
#----------------------- calculate  S_hat_x_A_to_B -------------------------------
ScA = Sx_generator(c_A, c_A)
ScB = Sx_generator(c_B, c_B)
S_hat_x_A2B = sum(alpha * (ScA - ScB), axis = 1)

#----------------------- calculate  distinctness -------------------------------
def distinctness (Sx_A2B, S_hat_x_A2B):
    return (1 - stats.kendalltau(Sx_A2B, S_hat_x_A2B))/2

distinctness (Sx_A2B, S_hat_x_A2B)

#----------------------- calculate  uncertainty -------------------------------
K_new = 2
p = np.zeros(batch_size, K_new)

def uncertainty(p):
    return(-1 * sum(p * (1 - p)))
