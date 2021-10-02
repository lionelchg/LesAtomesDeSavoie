import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from itertools import product

def Q(N, k):
    diags = np.ones((1, N))
    return sparse.dia_matrix((diags, [k]), shape=(N, N)).todense()

def cost_function(x, N):
    rho_k = 0.0
    for k in range(N - 1):
        Qx = np.matmul(Q(N, k), x)
        xQ = np.matmul(x, Q(N, k))
        rho_k += np.dot(Qx, x)**2

    return rho_k

if __name__ == '__main__':
    N = 5
    x = np.array([1, 1, 1, -1, 1])
    all_conf = np.array([np.array(i) for i in product([-1, 1], repeat=N)])
    for i, conf in enumerate(all_conf):
        print(cost_function(conf, N))