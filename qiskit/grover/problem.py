from qiskit_textbook.problems import grover_problem_oracle
from qiskit import QuantumCircuit

from utils import create_dir
from circuit import grover
from run import run_sim

if __name__ == '__main__':
    ## Example Usage of problem oracle
    n = 4
    oracle = grover_problem_oracle(n, variant=1)  # 0th variant of oracle, with n qubits
    qc = QuantumCircuit(n)

    # Construct grover circuit
    grover_circuit = grover(n, 1, oracle)

    # Run on simulator
    run_sim(grover_circuit, 'figures/hist_4bits_sim')
