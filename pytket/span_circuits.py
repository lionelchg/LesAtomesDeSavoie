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

from utils import create_dir, draw_circ

def optimize_circuit(circ:Circuit, seq_list:list, figname=None) -> None:
    """ Optimize in place the circuit

    :param circ: original circuit
    :type circ: Circuit
    :param seq_list: sequence for optimizer
    :type seq_list: list
    :param figname: name of the figure if print else None
    :type figname: str
    """
    # Draw original circuit
    if figname is not None:
        draw_circ(circ, f'{figname}_base')

    # Sequence of passes for optimization
    to_zx_optimised = SequencePass(seq_list)
    to_zx_optimised.apply(circ)
    
    # Draw optimized circuit
    if figname is not None:
        draw_circ(circ, f'{figname}_optim')

def params_circ(circ:Circuit):
    # Retrieve parameters of circuit
    depth = circ.depth()
    n_gates = circ.n_gates
    n_twogates = (circ.n_gates_of_type(OpType.CZ) + circ.n_gates_of_type(OpType.CX)
        + circ.n_gates_of_type(OpType.CY) + circ.n_gates_of_type(OpType.H))
    return depth, n_gates, n_twogates
    
if __name__ == '__main__':
    fig_dir = 'figures/optim/'
    create_dir(fig_dir)
    xscale = 'linear'
    
    circuit_dir = 'circuits/'
    circuits_fn = [f'small_{i:d}' for i in range(1, 5)] + [f'medium_{i:d}' for i in range(1, 5)]
    circuits_fn += [f'large_{i:d}' for i in range(1, 5)]

    ncircuits = len(circuits_fn)

    optims = dict()

    optims['Base'] = []
    # optims['Quil'] = [DecomposeBoxes(), RebaseQuil(), EulerAngleReduction(OpType.Rx, OpType.Rz)]
    # optims['IBM'] = [DecomposeBoxes(), RebaseIBM(), EulerAngleReduction(OpType.Rx, OpType.Rz)]
    # optims['Circ'] = [DecomposeBoxes(), RebaseCirq(), EulerAngleReduction(OpType.Rx, OpType.Rz)]

    df_depth = pd.DataFrame(np.zeros((ncircuits, len(optims))), index=circuits_fn, columns=optims.keys())
    df_gates = pd.DataFrame(np.zeros((ncircuits, len(optims))), index=circuits_fn, columns=optims.keys())
    df_twoq_gates = pd.DataFrame(np.zeros((ncircuits, len(optims))), index=circuits_fn, columns=optims.keys())

    # Loop on optims
    for optim, seq_list in optims.items():
        optim_figdir = f'{fig_dir}{optim}/'
        create_dir(optim_figdir)
        # Loop on circuits
        for circuit_fn in circuits_fn:
            print(f'{optim} + {circuit_fn}')
            figname = f'{optim_figdir}{circuit_fn}'

            # Creation of circuit from filename
            circ = circuit_from_qasm(f'{circuit_dir}{circuit_fn}.qasm')
            optimize_circuit(circ, seq_list, figname)

            # Retrieve parameters of circuit
            depth, n_gates, n_twogates = params_circ(circ)

            # Store the parameters into DataFrame
            df_depth[optim][circuit_fn] = depth
            df_gates[optim][circuit_fn] = n_gates
            df_twoq_gates[optim][circuit_fn] = n_twogates

    print('Depth:\n', df_depth)
    print('Gates:\n', df_gates)
    print('Two Gates:\n', df_twoq_gates)

    fig, ax = plt.subplots()
    df_depth.plot.barh(ax=ax)
    ax.set_xscale(xscale)
    ax.set_title('Circuit depth')
    fig.savefig(f'{optim_figdir}depth_base', bbox_inches='tight')

    fig, ax = plt.subplots()
    df_gates.plot.barh(ax=ax)
    ax.set_xscale(xscale)
    ax.set_title('Number of gates')
    fig.savefig(f'{optim_figdir}gates_base', bbox_inches='tight')

    fig, ax = plt.subplots()
    df_twoq_gates.plot.barh(ax=ax)
    ax.set_xscale(xscale)
    ax.set_title('Number of two qubits gates')
    fig.savefig(f'{optim_figdir}twogates_base', bbox_inches='tight')