import numpy as np
from qiskit import QuantumCircuit, execute, Aer

if __name__ == '__main__':
    # Circuit to construct Bell state
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    # Execute and get results
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print("Total counts for 00 and 11 are ", counts)