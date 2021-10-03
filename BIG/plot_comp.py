import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def compute_rho(chain):
    result = []
    chain = [2 * int(char) - 1 for char in chain]
    for tau in range(1, len(chain)+1):
        sumbin = 0
        for i in range(tau):
            sumbin += chain[i] * chain[i+len(chain)-tau]
        result.append(abs(sumbin))

    return result

def plot_chain(chain: str, chain_th: str, figname: Path):
    result = compute_rho(chain)
    result_th = compute_rho(chain_th)

    x = np.arange(1, len(chain)+1)
    fig, ax = plt.subplots()
    ax.set_title("Autocorrelation shape")
    ax.set_xlabel("Delay")
    ax.set_ylabel("Energy level")
    ax.plot(x, result, '--', color ="red", label='Experiment')
    ax.plot(x, result_th, '-.', color ="blue", label='Barker')
    ax.grid(True)
    ax.legend()
    fig.savefig(figname)

if __name__ == '__main__':
    chain_res = '1000110011011'
    chain_barker = '0000011001010'
    plot_chain(chain_res, chain_barker, '13_exp_vs_barker')