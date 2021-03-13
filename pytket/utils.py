import os
import matplotlib.pyplot as plt
from pytket.circuit import Circuit
from pytket.extensions.qiskit import tk_to_qiskit

def create_dir(dir_name:str) -> None:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def draw_circ(circ:Circuit, filename:str):
    qk_circ = tk_to_qiskit(circ)
    qk_circ.draw(output='mpl', filename=filename)
    plt.close()