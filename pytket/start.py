from pytket import Circuit

if __name__ == '__main__':
    n_qubits = 3
    circ = Circuit(n_qubits)

    # une rotation X sur le qubit 1
    circ.Rx(1)
    # une rotation Z sur le qubit 0
    circ.Rz(0)
    # une porte CNot entre qubits 0 et 2
    circ.CX(0, 2)

