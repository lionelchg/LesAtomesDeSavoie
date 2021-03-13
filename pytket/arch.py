from pytket.routing import Architecture
from pytket.circuit import Node, Circuit
from pytket.device import Device
from pytket.passes import DefaultMappingPass, DecomposeSwapsToCXs
from pytket.predicates import CompilationUnit

from utils import draw_circ, create_dir

if __name__ == '__main__':
    fig_dir = 'figures/arch/'
    create_dir(fig_dir)

    # Create linear 5-qubit architecture
    n = [Node('n', i) for i in range(5)]
    arc = Architecture([[n[0], n[1]], [n[1], n[2]], 
                        [n[2], n[3]], [n[3], n[4]]])
    dev = Device(arc)

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

    draw_circ(circ, f'{fig_dir}circ_base')

    # Map this circuit for our device linked
    # to the 5-qubit architecture
    mapper = DefaultMappingPass(dev)
    cu = CompilationUnit(circ)
    mapper.apply(cu)
    circ1 = cu.circuit
    draw_circ(circ1, f'{fig_dir}circ_dev')

    pass1 = DecomposeSwapsToCXs(dev)
    pass1.apply(cu)
    circ2 = cu.circuit
    draw_circ(circ2, f'{fig_dir}circ_decompose')
