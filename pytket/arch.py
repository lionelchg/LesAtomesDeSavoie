from pytket.routing import Architecture
from pytket.circuit import Node, Circuit
from pytket.backends.backendinfo import BackendInfo
from pytket.passes import DefaultMappingPass, DecomposeSwapsToCXs
from pytket.predicates import CompilationUnit

from utils import draw_circ
from pathlib import Path

if __name__ == '__main__':
    fig_dir = Path('figures/arch/')
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Create linear 5-qubit architecture
    n = [Node('n', i) for i in range(5)]
    arc = Architecture([[n[0], n[1]], [n[1], n[2]],
                        [n[2], n[3]], [n[3], n[4]]])
    dev = BackendInfo('Backtest', 'Linearmap', architecture=arc)

    # Create circuit
    circ = Circuit(5)
    circ.CX(0, 1)
    circ.H(0)
    circ.Z(1)
    circ.CX(0, 3)
    circ.Rx(1.5, 3)
    circ.CX(2, 4)
    circ.X(2)
    circ.CX(1, 4)
    circ.CX(0, 4)

    draw_circ(circ, fig_dir / 'circ_base')

    # Map this circuit for our device linked
    # to the 5-qubit architecture
    mapper = DefaultMappingPass(dev)
    cu = CompilationUnit(circ)
    mapper.apply(cu)
    circ1 = cu.circuit
    draw_circ(circ1, fig_dir / 'circ_dev')

    pass1 = DecomposeSwapsToCXs(dev)
    pass1.apply(cu)
    circ2 = cu.circuit
    draw_circ(circ2, fig_dir / 'circ_decompose')
