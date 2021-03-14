## import the necessary tools for our work
from IPython.display import clear_output
from  qiskit import *
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq import least_busy
from qiskit.test.mock import FakeAlmaden, FakeMelbourne, FakeManhattan, FakeMontreal

# Visualisation settings
from plot import ax_prop
import seaborn as sns, operator

sns.set_style("whitegrid")
sns.set_context("talk")

## Code for inverse Quantum Fourier Transform
## adapted from Qiskit Textbook at
## qiskit.org/textbook

def qft_dagger(circ_, n_qubits):
    """n-qubit QFTdagger the first n qubits in circ"""
    for qubit in range(int(n_qubits/2)):
        circ_.swap(qubit, n_qubits-qubit-1)
    for j in range(0,n_qubits):
        for m in range(j):
            circ_.cp(-np.pi/float(2**(j-m)), m, j)
        circ_.h(j)

## Code for initial state of Quantum Phase Estimation
## adapted from Qiskit Textbook at qiskit.org/textbook
## Note that the starting state is created by applying 
## H on the first n_qubits, and setting the last qubit to |psi> = |1>

def qpe_pre(circ_, n_qubits):
    circ_.h(range(n_qubits))
    circ_.x(n_qubits)

    for x in reversed(range(n_qubits)):
        for _ in range(2**(n_qubits-1-x)):
            circ_.cp(1, n_qubits-1-x, n_qubits)

## Run a Qiskit job on either hardware or simulators

def run_job(circ, backend, shots=1000, optimization_level=0):
    t_circ = transpile(circ, backend, optimization_level=optimization_level)
    qobj = assemble(t_circ, shots=shots)
    job = backend.run(qobj)
    job_monitor(job)
    return job.result().get_counts()

## Function to estimate pi
## Summary: using the notation in the Qiskit textbook (qiskit.org/textbook),
## do quantum phase estimation with the 'phase' operator U = p(theta) and |psi> = |1>
## such that p(theta)|1> = exp(2 x pi x i x theta)|1>
## By setting theta = 1 radian, we can solve for pi
## using 2^n x 1 radian = most frequently measured count = 2 x pi

def get_pi_estimate(n_qubits, simulator, print_circ=False):

    # create the circuit
    circ = QuantumCircuit(n_qubits + 1, n_qubits)
    # create the input state
    qpe_pre(circ, n_qubits)
    # apply a barrier
    circ.barrier()
    # apply the inverse fourier transform
    qft_dagger(circ, n_qubits)
    # apply  a barrier
    circ.barrier()
    # measure all but the last qubits
    circ.measure(range(n_qubits), range(n_qubits))

    if print_circ: circ.draw(output='mpl', filename=f'figures/pi_circ_{n_qubits:d}')

    # run the job and get the results
    counts = run_job(circ, backend=simulator, shots=1000, optimization_level=0)
    # print(counts) 

    # get the count that occurred most frequently
    max_counts_result = max(counts, key=counts.get)
    max_counts_result = int(max_counts_result, 2)
    
    # solve for pi from the measured counts
    theta = max_counts_result/2**n_qubits
    return (1. / (2 * theta))

def estimate_pi(n_qubits_max, simulator, casename):
    # estimate pi using different numbers of qubits
    pi = np.pi
    nqs = list(range(2, n_qubits_max + 1, 1))
    pi_estimates = []
    for nq in nqs:
        thisnq_pi_estimate = get_pi_estimate(nq, simulator)
        pi_estimates.append(thisnq_pi_estimate)
        print(f"{nq} qubits, pi ≈ {thisnq_pi_estimate}")
    
    array_estimates = np.zeros((len(nqs), 2))
    array_estimates[:, 0] = np.array(nqs)
    array_estimates[:, 1] = np.array(pi_estimates)
    np.save(f'data/{casename}', array_estimates)

    fig, ax = plt.subplots()
    ax.plot(nqs, [pi]*len(nqs), '--r', label='$\pi$')
    ax.plot(nqs, pi_estimates, '.-', label='Approx $\pi$')
    ax_prop(ax, [nqs[0] - 0.5, nqs[-1] + 0.5], [1.5, 4.5], casename)
    plt.savefig(f'figures/{casename}', bbox_inches='tight')

if __name__ == '__main__':
    # Local simulator
    simulator = Aer.get_backend('qasm_simulator')
    simulator_realdevice = FakeMontreal()

    # Load your IBMQ account if 
    # you'd like to use the cloud simulator or real quantum devices
    my_provider = IBMQ.load_account()
    simulator_cloud = my_provider.get_backend('ibmq_qasm_simulator')
    device = my_provider.get_backend('ibmq_16_melbourne')

    print("Running on device: ", simulator_realdevice)

    # provider = IBMQ.get_provider("ibm-q")
    # device = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 3 and 
    #                                 not x.configuration().simulator and x.status().operational==True))
    # print("Running on current least busy device: ", device)

    estimate_pi(10, simulator, 'local_qasm')
    estimate_pi(10, simulator_cloud, 'ibmq_qasm')
    estimate_pi(10, simulator_realdevice, str(simulator_realdevice))
    estimate_pi(10, device, str(device))