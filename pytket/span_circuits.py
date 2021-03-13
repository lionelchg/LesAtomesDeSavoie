import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pytket.qasm import circuit_from_qasm
from pytket.passes import (RebaseCirq, RebaseCustom, RebaseHQS, 
                        RebaseIBM, RebaseProjectQ, RebasePyZX, 
                        RebaseQuil, RebaseTket)
from pytket.passes import (CliffordSimp, SynthesiseIBM, SequencePass, 
        DecomposeBoxes, FullPeepholeOptimise, EulerAngleReduction)
from pytket.circuit import Circuit, Unitary1qBox, Unitary2qBox, OpType, Op
from pytket.extensions.qiskit import tk_to_qiskit

from utils import create_dir, fig_dir

def draw_circ(circ:Circuit, filename:str):
    qk_circ = tk_to_qiskit(circ)
    qk_circ.draw(output='mpl', filename=filename)

def convert_circuit(circuit_dir:str, circuit_fn:str, seq_list:list, 
            figname:str, lprint:bool=False):
    circ = circuit_from_qasm(f'{circuit_dir}{circuit_fn}.qasm')

    if lprint:
        draw_circ(circ, f'{figname}_base')

    to_zx_optimised = SequencePass(seq_list)

    # we now use the same pass as before:
    to_zx_optimised.apply(circ)
    
    if lprint:
        draw_circ(circ, f'{figname}_optim')

    return circ.depth(), circ.n_gates

if __name__ == '__main__':
    circuit_dir = 'circuits/'
    circuits_fn = [f'small_{i:d}' for i in range(1, 5)] + [f'medium_{i:d}' for i in range(1, 5)]
    ncircuits = len(circuits_fn)

    optims = dict()

    optims['Quil'] = [DecomposeBoxes(), RebaseQuil(), EulerAngleReduction(OpType.Rx, OpType.Rz)]
    optims['IBM'] = [DecomposeBoxes(), RebaseIBM(), EulerAngleReduction(OpType.Rx, OpType.Rz)]
    optims['Circ'] = [DecomposeBoxes(), RebaseCirq(), EulerAngleReduction(OpType.Rx, OpType.Rz)]

    df_depth = pd.DataFrame(np.zeros((ncircuits, 3)), index=circuits_fn, columns=optims.keys())
    df_gates = pd.DataFrame(np.zeros((ncircuits, 3)), index=circuits_fn, columns=optims.keys())

    for optim, seq_list in optims.items():
        optim_figdir = f'{fig_dir}{optim}/'
        create_dir(optim_figdir)
        for circuit_fn in circuits_fn:
            figname = f'{optim_figdir}{circuit_fn}'
            depth, ngates = convert_circuit(circuit_dir, circuit_fn, 
                    seq_list, figname, lprint=True)
            df_depth[optim][circuit_fn] = depth
            df_gates[optim][circuit_fn] = ngates

    print(df_depth)
    print(df_gates)