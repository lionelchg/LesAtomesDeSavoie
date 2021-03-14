import matplotlib.pyplot as plt

def ax_prop(ax, xlim, ylim, figname):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend()
    ax.set_xlabel('Number of qubits')
    ax.set_ylabel('$\pi$ and estimate of $\pi$')
    ax.set_title(figname)