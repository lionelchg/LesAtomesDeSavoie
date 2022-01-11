import numpy as np
import itertools
import matplotlib.pyplot as plt
from tqdm import *
from pathlib import Path
from scipy.sparse import diags

fig_dir = Path('figures')
fig_dir.mkdir(parents=True, exist_ok=True)


def generate_Q_list(N):
    """ Generate list of matrices of
    to offset the signal (compute x_{i+k}) """
    Q_list = []
    for k in range(1, N+1):
        Q_list.append(np.diagflat(np.ones(k), N - k))
    return Q_list


def return_fig(conf, Q_list, ax):
    """ Print figure of the signal """
    res = []
    for Q in Q_list:
        res.append(conf.dot(Q.dot(conf.T)))
    ax.plot(res)


def cost(N):
    """ Compute the cost in brut force for the
    Waveform Optimization problem """
    # Compute the list of diagonal matrices for cost function computation
    Q_list = generate_Q_list(N=N)

    # Generate all the [-1, 1] configurations available
    all_conf = np.array(list(itertools.product([-1, 1], repeat=N)))

    # Compute the costs
    cost = np.zeros(2**N)
    for Q in Q_list:
        for k, x in tqdm(enumerate(all_conf)):
            cost[k] += np.abs(x.dot(Q.dot(x.T)))
    cost = np.array(cost)

    # Only take the first example
    ind = np.where(cost == cost.min())

    # Plot the minimum
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
