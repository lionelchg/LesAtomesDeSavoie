import numpy as np

from pytket.qasm import circuit_from_qasm
from pytket.passes import (RebaseCirq, RebaseCustom, RebaseHQS, 
                        RebaseIBM, RebaseProjectQ, RebasePyZX, 
                        RebaseQuil, RebaseTket)
from pytket.passes import (CliffordSimp, SynthesiseIBM, SequencePass, 
        DecomposeBoxes, FullPeepholeOptimise, EulerAngleReduction)
from pytket.circuit import Circuit, Unitary1qBox, Unitary2qBox, OpType, Op
from pytket.extensions.qiskit import tk_to_qiskit

from utils import fig_dir

def draw_circ(circ:Circuit, filename:str):
    qk_circ = tk_to_qiskit(circ)
    qk_circ.draw(output='mpl', filename=filename)

def convert_circuit(circuit_fn:str, seq_list:list, lprint:bool=False):
    circ = circuit_from_qasm(circuit_fn)

    if lprint:
        draw_circ(circ, f'{fig_dir}{circuit_fn}_base')

    to_zx_optimised = SequencePass(seq_list)

    # we now use the same pass as before:
    to_zx_optimised.apply(circ)
    
    if lprint:
        draw_circ(circ, f'{fig_dir}{circuit_fn}_opti')

    return circ.depth(), circ.n_gates

if __name__ == '__main__':
    circuit_dir = 'circuits/'
    circuits_fn = [f'small_{i:d}.qasm' for i in range(1, 5)] + [f'medium_{i:d}.qasm' for i in range(1, 5)]
    seq_list = [
        DecomposeBoxes(),
        RebaseQuil(),
        EulerAngleReduction(OpType.Rx, OpType.Rz)
    ]

    for circuit_fn in circuits_fn:
        print(f'Decomposing: {circuit_fn}')
        depth, n_gates = convert_circuit(circuit_dir + circuit_fn, seq_list, lprint=True)
        print(f'depth = {depth:d} - n_gates = {n_gates:d}')