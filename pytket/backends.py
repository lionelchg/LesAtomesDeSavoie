from pytket.extensions.qiskit import AerStateBackend, AerBackend, AerUnitaryBackend, IBMQBackend, IBMQEmulatorBackend
from pytket.extensions.projectq import ProjectQBackend
from pytket import Circuit
from qiskit.providers.aer.noise import NoiseModel
from pytket.utils import counts_from_shot_table

if __name__ == '__main__':
    circ = Circuit(2)
    circ.H(0)
    circ.CX(0, 1)

    aer_state_b = AerStateBackend()
    state_handle = aer_state_b.process_circuit(circ)
    statevector = aer_state_b.get_result(state_handle).get_state()
    print(statevector)

    aer_unitary_b = AerUnitaryBackend()
    print(aer_unitary_b.get_unitary(circ))

    circ.measure_all()

    aer_b = AerBackend()
    shots_handle = aer_b.process_circuit(circ, n_shots=10, seed=1)

    shots = aer_b.get_result(shots_handle).get_shots()
    print(shots)
    print(counts_from_shot_table(shots))

    my_noise_model = NoiseModel()
    readout_error = 0.2
    for q in range(2):
        my_noise_model.add_readout_error([[1-readout_error, readout_error], [readout_error, 1-readout_error]], [q])

    noisy_aer_b = AerBackend(my_noise_model)
    noisy_shots_handle = noisy_aer_b.process_circuit(circ, n_shots=10, seed=1, valid_check=False)
    noisy_shots = noisy_aer_b.get_result(noisy_shots_handle).get_shots()
    print(noisy_shots)
    print(counts_from_shot_table(noisy_shots))