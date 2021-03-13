from pytket import Circuit
from pytket.extensions.qiskit import AerBackend # le simulateur

if __name__ == '__main__':
    # défini le circuit comme dans la section précédente
    circ = Circuit(2, 2)
    circ.Rx(0.3, 0).Ry(0.5, 1).CRz(-0.6, 1, 0).measure_all()

    # initialise le simulateur
    backend = AerBackend()
    # chaque machine et simulateur a des restrictions concernant
    # les portes que peuvent être exécutées
    # la compilation du circuit guaranti que le circuit est dans un format
    # reconnu par simulateur (ou l'ordinateur quantique)
    backend.compile_circuit(circ)
    # exécute la simulation
    handle = backend.process_circuit(circ, n_shots=20)
    # obtient les résultats
    shots = backend.get_result(handle).get_shots()