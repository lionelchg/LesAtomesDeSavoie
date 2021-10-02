import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from qiskit.visualization import plot_histogram
from pathlib import Path

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, execute
from qiskit.circuit import Parameter

def cost_function(x, nqubits):
    """
    Given a bitstring as a solution, this function returns
    the number of edges shared between the two partitions
    of the graph.

    Args:
        x: str
           solution bitstring

        G: networkx graph

    Returns:
        obj: float
             Objective
    """
    obj = 0

    for k in range(1,nqubits):
        tmp = 0
        for i in range(1,nqubits-k+1):
            tmp += (2*int(x[i-1])-1)*(2*int(x[i+k-1])-1)
        obj += tmp**2


    return obj


def compute_expectation(counts,  nqubits):

    """
    Computes expectation value based on measurement results

    Args:
        counts: dict
                key as bitstring, val as count

        G: networkx graph

    Returns:
        avg: float
             expectation value
    """

    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():

        obj = cost_function(x=bitstring, nqubits=nqubits)
        avg += obj * count
        sum_count += count

    return avg/sum_count


# We will also bring the different circuit components that
# build the qaoa circuit under a single function
def create_qaoa_circ(theta,nqubits):

    """
    Creates a parametrized qaoa circuit

    Args:
        G: networkx graph
        theta: list
               unitary parameters

    Returns:
        qc: qiskit circuit
    """
    p = int(len(theta)/2)
    #p = len(theta)//2  # number of alternating unitaries
    qc = QuantumCircuit(nqubits)

    beta = theta[:p]
    gamma = theta[p:]
    #print(theta,beta,gamma)

    # initial_state
    for i in range(0, nqubits):
        qc.h(i)

    for irep in range(0, p):

        fourfactors = np.exp(-1j * 2 * gamma[irep] * np.array([1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1]))

        # problem unitary
        for k in range(1,nqubits):
            for i in range(0, nqubits-k):
                qc.rzz(4 * gamma[irep], i, i+k)
            for i in range(0, nqubits-k):
                for j in range(i + 1, nqubits-k):
                    if i + k != j:
                        qc.diagonal(list(fourfactors), qubit=[i, i+k, j, j+k])


        # mixer unitary
        for i in range(0, nqubits):
            qc.rx(2 * beta[irep], i)

    qc.measure_all()

    return qc

# Finally we write a function that executes the circuit on the chosen backend
def get_expectation(nqubits, shots=4096):

    """
    Runs parametrized circuit

    Args:
        G: networkx graph
        p: int,
           Number of repetitions of unitaries
    """
    shots = max(shots, 2**nqubits)
    backend = Aer.get_backend('qasm_simulator')
    backend.shots = shots

    def execute_circ(theta):

        qc = create_qaoa_circ(theta=theta,nqubits=nqubits)
        counts = backend.run(circuits = qc, seed_simulator=1,
                             nshots=shots).result().get_counts()

        return compute_expectation(counts=counts,nqubits=nqubits)

    return execute_circ

if __name__ == '__main__':
    fig_dir = Path('figures/qaoa')
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Scipy minimize methods
    minimize_methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'COBYLA']

    # Define problem
    nqubits = 5
    expectation = get_expectation(nqubits=nqubits)
    p0 = np.random.uniform(size = 20)
    res = minimize(expectation,
                    x0 = p0,
                method='COBYLA')
    print(res)

    shots = 4096
    backend = Aer.get_backend('aer_simulator')

    qc_res = create_qaoa_circ(res.x,nqubits)

    counts = backend.run(qc_res, seed_simulator=10, shots = shots).result().get_counts()

    # Plot results after pproc
    fig, ax = plt.subplots()
    exp_cost = []
    exp_counts = []
    for key in counts.keys():
        exp_cost.append(cost_function(key, nqubits))
        exp_counts.append(counts[key])

    ax.scatter(exp_cost,exp_counts)
    ax.set_xscale('log')
    ax.set_xlabel('Cost')
    ax.set_ylabel('Counts')
    ax.grid(True)
    fig.savefig(fig_dir / f'N_{nqubits:d}_cost_counts_')