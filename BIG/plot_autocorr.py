import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_chain(chain: str, figname: Path):
    result = []
    chain = [2 * int(char) - 1 for char in chain]
    for tau in range(1, len(chain)+1):
        sumbin = 0
        for i in range(tau):
            sumbin += chain[i] * chain[i+len(chain)-tau]
        result.append(abs(sumbin))

    x = np.arange(1,len(chain)+1)
    fig, ax = plt.subplots()
    ax.set_title("Autocorrelation shape")
    ax.set_xlabel("Delay")
    ax.set_ylabel("Energy level")
    ax.plot(x, result, color ="red")
    ax.plot(x, np.zeros(len(chain)), color="blue")
    ax.grid(True)
    fig.savefig(figname)

if __name__ == '__main__':
    fig_dir = Path('figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    tmp_str = '000100'
    plot_chain(tmp_str, fig_dir / 'test')