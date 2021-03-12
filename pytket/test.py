from pytket import Circuit
from pytket.extensions.qiskit import IBMQBackend, AerStateBackend

if __name__ == '__main__':
    c = Circuit(2,2) # define a circuit with 2 qubits and 2 bits
    c.H(0)           # add a Hadamard gate to qubit 0
    c.Rz(0.25, 0)    # add an Rz gate of angle 0.25*pi to qubit 0
    c.CX(1,0)        # add a CX gate with control qubit 1 and target qubit 0
    c.measure_all()  # measure qubits 0 and 1, recording the results in bits 0 and 1

    sim_b = AerStateBackend()
    print(sim_b.required_predicates)
    # dev_b = IBMQBackend("ibm-q")
    # print(dev_b.required_predicates)

    initial_state = Circuit(3)      # Initialise the system in 1/sqrt(2) (|011> + |101>)
    initial_state.H(0)
    initial_state.X(1)
    initial_state.X(2)
    initial_state.CX(0, 1)

    increment = Circuit(3)
    increment.CCX(2, 1, 0)
    increment.CX(2, 1)
    increment.X(2)

    final_state = initial_state.copy()
    final_state.append(increment)

    from pytket.extensions.qiskit import AerStateBackend
    b = AerStateBackend()
    b.compile_circuit(initial_state)
    b.compile_circuit(final_state)
    handles = b.process_circuits([initial_state, final_state])
    s0 = b.get_state(handles[0])    # Check that the initial state is 1/sqrt(2) (|011> + |101>)
    print(s0.round(10))             # Round to ignore floating-point error in simulation
    s1 = b.get_state(handles[1])    # Check that the incrementer has mapped this to 1/sqrt(2) (|100> + |110>)
    print(s1.round(10))