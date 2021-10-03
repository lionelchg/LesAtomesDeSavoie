import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import itertools
import sympy

from pytket.pauli import Pauli
import pytket as tk
from pytket.extensions.qiskit import AerBackend, tk_to_qiskit

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

    for k in range(1, nqubits):
        tmp = 0
        for i in range(1, nqubits - k + 1):
            tmp += (2 * int(x[i - 1]) - 1) * (2 * int(x[i + k - 1]) - 1)
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
        print(bitstring, count)

        obj = cost_function(x=bitstring, nqubits=nqubits)
        avg += obj * count
        sum_count += count
    print(avg / sum_count)
    return avg / sum_count


# We will also bring the different circuit components that
# build the qaoa circuit under a single function
def create_qaoa_circ(beta, gamma, backend, nqubits):

    """
    Creates a parametrized qaoa circuit

    Args:
        G: networkx graph
        theta: list
               unitary parameters

    Returns:
        qc: qiskit circuit
    """
    # p = int(len(theta) / 2)
    qc = tk.Circuit(nqubits)

    # initial_state
    for i in range(0, nqubits):
        qc.H(i)

    for irep in range(0, len(beta)):

        pauli_tmp = tk.circuit.PauliExpBox([Pauli.Z, Pauli.Z, Pauli.Z, Pauli.Z], 2 * gamma[irep])

        # problem unitary
        for k in range(1,nqubits):
            for i in range(0, nqubits - 2 * k):
                qc.ZZPhase(4 * gamma[irep], i, i + 2 * k)
            for i in range(0, nqubits-k):
                for j in range(i + 1, nqubits-k):
                    if i + k != j:
                        qc.add_pauliexpbox(pauli_tmp, [i, i + k, j, j + k])

        # mixer unitary
        for i in range(0, nqubits):
            qc.Rx(2 * beta[irep], i)


    qc = backend.get_compiled_circuit(qc)
    qc.measure_all()

    return qc

# Finally we write a function that executes the circuit on the chosen backend
def get_expectation(template, beta, gamma,  backend, nqubits, shots=1024):

    """
    Runs parametrized circuit

    Args:
        G: networkx graph
        p: int,
           Number of repetitions of unitaries
    """
    shots = max(shots, 2**nqubits)
    n = len(beta)

    def execute_circ(theta):
        qc = template.copy()
        qc.symbol_substitution({
            beta[i]: theta[i] for i in range(len(beta))
        } )
        qc.symbol_substitution({
            gamma[i]: theta[i + n] for i in range(len(beta))
        })
        # qc = create_qaoa_circ(theta=theta,nqubits=nqubits)
        handle = backend.process_circuit(qc, n_shots=shots)
        counts = backend.get_result(handle).get_counts()
        print(counts)

        return compute_expectation(counts=counts,nqubits=nqubits)

    return execute_circ

def run_exp(nqubits: int, opt_method: str, fig_dir: Path):
    """ Run experiment of QAOA """
    fig_dir = fig_dir / opt_method
    fig_dir.mkdir(parents=True, exist_ok=True)

    nparams = 2
    beta = sympy.symbols(f'b_0:{nparams}')
    gamma = sympy.symbols(f'g_0:{nparams}')
    backend = AerBackend()
    template = create_qaoa_circ(beta, gamma, backend, nqubits)

    expectation = get_expectation(template, beta, gamma, backend, nqubits=nqubits)
    p0 = np.random.uniform(size = 2 * nparams)
    res = minimize(expectation,
                    x0 = p0,
                method=opt_method,
                options={'maxiter': 50})

    if res.success:
        print('Convergence OK')
    else:
        print('Convergence failed')

    shots = 1024

    qc_res = template.copy()
    qc_res.symbol_substitution({
            beta[i]: res.x[i] for i in range(nparams)
        } )
    qc_res.symbol_substitution({
            gamma[i]: res.x[i + nparams] for i in range(nparams)
        })
    backend = AerBackend()
    qc_res = backend.get_compiled_circuit(qc_res)
    handle = backend.process_circuit(qc_res, n_shots=shots)
    counts = backend.get_result(handle).get_counts()

    qc_qiskit = tk_to_qiskit(qc_res)
    qc_qiskit.draw(output='mpl', filename=fig_dir / f'circuit_N{nqubits:d}')

    # Number the chosen bitstrings
    array_bitstrings = np.array(list(counts.keys()))
    array_counts = np.array(list(counts.values()))
    list_ind = np.argwhere(array_counts > 0.6 * np.max(array_counts))

    # Seleceted bitstrings
    selected_bitstrings = array_bitstrings[list_ind]
    selected_counts = array_counts[list_ind]
    print('| Selected string | Count |')
    tmp_dir = fig_dir / f'N_{nqubits:d}'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for istring, (selected_bigstring, selected_count) in enumerate(zip(selected_bitstrings, selected_counts)):
        print(selected_bigstring, selected_count[0])
        plot_chain(selected_bigstring[0], tmp_dir / f'auto_corr_{istring:d}')

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
    fig.savefig(fig_dir / f'N_{nqubits:d}_cost_counts')

def plot_chain(chain: str, figname: Path):
    result = []
    chain = [2 * int(char) - 1 for char in chain]
    for tau in range(1, len(chain)+1):
        sumbin = 0
        for i in range(tau):
            sumbin += chain[i] * chain[i+len(chain)-tau]
        result.append(abs(sumbin))

    x = np.arange(1, len(chain)+1)
    fig, ax = plt.subplots()
    ax.set_title("Autocorrelation shape")
    ax.set_xlabel("Delay")
    ax.set_ylabel("Energy level")
    ax.plot(x, result, color ="red")
    ax.plot(x, np.zeros(len(chain)), color="blue")
    ax.grid(True)
    fig.savefig(figname)

if __name__ == '__main__':
    fig_dir = Path('figures/qaoa/pytket')
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Scipy minimize methods
    nqubits_list = [3, 4, 5, 6, 7]
    # minimize_methods = ['Nelder-Mead', 'Powell', 'COBYLA']
    minimize_methods = ['Nelder-Mead']

    params_list = list(itertools.product(nqubits_list, minimize_methods))

    # Define problem
    for params in params_list:
        run_exp(*params, fig_dir)