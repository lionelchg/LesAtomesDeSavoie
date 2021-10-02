import numpy as np
import itertools
import matplotlib.pyplot as plt
from tqdm import *
from pathlib import Path

fig_dir = Path('figures')
fig_dir.mkdir(parents=True, exist_ok=True)

def generate_Q(N = 5, k = 1):
    Q = np.ones((N, N))
    Q_ = -np.ones((N,N))
    if k > 1:
        return np.triu(Q,k=N-k)+np.triu(Q_,k=N-k+1)
    else:
        return np.triu(Q,k=N-k)
def generate_Q_list(N):
    Q_list = []
    for k in range(1,N+1):
        Q_list.append(generate_Q(N=N,k=k))
    return Q_list

def return_fig(conf, Q_list, ax):
    res = []
    for Q in Q_list:
        res.append(conf.dot(Q.dot(conf.T)))
    ax.plot(res)

def cost(N):
    Q_list = generate_Q_list(N = N)
    all_conf = np.array([np.array(i) for i in itertools.product([-1, 1], repeat = N)])
    cost = np.zeros(2**N)
    for Q in Q_list:
        for k, x in tqdm(enumerate(all_conf)):
            cost[k] += np.abs(x.dot(Q.dot(x.T)))
    cost = np.array(cost)
    ind = np.where(cost == cost.min())

    fig, ax = plt.subplots()
    for conf in all_conf[ind]:
        return_fig(conf, Q_list, ax)
    fig.savefig(fig_dir / f'optim_{N:d}')
    plt.close(fig)

    return all_conf[ind]

if __name__ == '__main__':
    list_N = [5, 11, 15]
    for N in list_N:
        cost(N)