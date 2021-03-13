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
    plt.close()

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
    
    n_twogates = (circ.n_gates_of_type(OpType.CZ) + circ.n_gates_of_type(OpType.CX)
                + circ.n_gates_of_type(OpType.CY) + circ.n_gates_of_type(OpType.H))

    return circ.depth(), circ.n_gates, n_twogates

if __name__ == '__main__':
    circuit_dir = 'circuits/'
    circuits_fn = [f'small_{i:d}' for i in range(1, 5)] + [f'medium_{i:d}' for i in range(1, 5)]
    circuits_fn += [f'large_{i:d}' for i in range(1, 5)]

    ncircuits = len(circuits_fn)

    optims = dict()

    optims['Quil'] = [DecomposeBoxes(), RebaseQuil(), EulerAngleReduction(OpType.Rx, OpType.Rz)]
    optims['IBM'] = [DecomposeBoxes(), RebaseIBM(), EulerAngleReduction(OpType.Rx, OpType.Rz)]
    optims['Circ'] = [DecomposeBoxes(), RebaseCirq(), EulerAngleReduction(OpType.Rx, OpType.Rz)]

    df_depth = pd.DataFrame(np.zeros((ncircuits, 3)), index=circuits_fn, columns=optims.keys())
    df_gates = pd.DataFrame(np.zeros((ncircuits, 3)), index=circuits_fn, columns=optims.keys())
    df_twoq_gates = pd.DataFrame(np.zeros((ncircuits, 3)), index=circuits_fn, columns=optims.keys())

    for optim, seq_list in optims.items():
        optim_figdir = f'{fig_dir}{optim}/'
        create_dir(optim_figdir)
        for circuit_fn in circuits_fn:
            print(f'{optim} + {circuit_fn}')
            figname = f'{optim_figdir}{circuit_fn}'
            depth, ngates, ntwogates = convert_circuit(circuit_dir, circuit_fn, 
                    seq_list, figname)
            df_depth[optim][circuit_fn] = depth
            df_gates[optim][circuit_fn] = ngates
            df_twoq_gates[optim][circuit_fn] = ntwogates

    print('Depth:\n', df_depth)
    print('Gates:\n', df_gates)
    print('Two Gates:\n', df_twoq_gates)

    fig, ax = plt.subplots()
    df_depth.plot.barh(ax=ax)
    ax.set_title('Circuit depth')
    fig.savefig('figures/depth', bbox_inches='tight')

    fig, ax = plt.subplots()
    df_gates.plot.barh(ax=ax)
    ax.set_title('Number of gates')
    fig.savefig('figures/gates', bbox_inches='tight')

    fig, ax = plt.subplots()
    df_twoq_gates.plot.barh(ax=ax)
    ax.set_title('Number of two qubits gates')
    fig.savefig('figures/twogates', bbox_inches='tight')