#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 10:43:41 2021

@author: bruno
"""

import numpy as np
from scipy.linalg import toeplitz
from geneticalgorithm import geneticalgorithm as ga

N = 5
Q = 2
R = np.zeros((N, N, N))

R[:, :, 0] = np.identity(N)
for n in range(1, N):
    R[:, :, n] = toeplitz(np.zeros(N), np.concatenate((np.zeros(n), np.ones(1), np.zeros(N-n-1))))

def rho(x, y, R, k):
    if k >= 0:
        return x.T@R[:, :, k]@y
    else:
        return y.T@R[:, :, -k]@x

model = ga(function=islr, dimension=2*N, variable_type='int', variable_boundaries=np.array([[0,Q-1]]*2*N))

model.run()