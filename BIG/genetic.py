import numpy as np
from scipy.linalg import toeplitz
from geneticalgorithm import geneticalgorithm as ga

def rho(x, y, R, k):
    return x.T@R[:, :, k]@y

def islr(phi):
    x = np.exp(-1j*2*np.pi*phi/Q)
    ssq = 0
    for k in range(R.shape[2]):
        ssq = ssq + np.abs(rho(x, x, R, k)) ** 2
    return ssq

if __name__ == '__main__':
    N = 13
    Q = 2
    R = np.zeros((N, N, N))
    R[:, :, 0] = np.identity(N)
    for n in range(1, N):
        R[:, :, n] = toeplitz(np.zeros(N), np.concatenate((np.zeros(n), np.ones(1), np.zeros(N - n - 1))))

    model = ga(function=islr, dimension=N, variable_type='int', variable_boundaries=np.array([[0, Q - 1]] * N))
    model.run()