
from pytket.circuit import Node, Circuit
from pytket.extensions.qiskit import AerBackend # the simulator

# defined the circuit as in the previous section
circ = Circuit(2, 2)
circ.Rx(0.3, 0).Ry(0.5, 1).CRz(-0.6, 1, 0).measure_all()

# initializes the simulator
backend = AerBackend()
# each machine and simulator have restrictions regarding
# the gates that can be executed
# the compilation of the circuit guarantees the circuit is in a format
# recognized by simulator (or quantum computer)
backend.compile_circuit(circ)
# runs the simulation
handle = backend.process_circuit(circ, n_shots=20)
# gets the results
shots = backend.get_result(handle).get_shots()
print(shots)