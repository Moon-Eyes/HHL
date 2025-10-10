import numpy as np
from qiskit import QuantumCircuit, transpile
import torch
from qiskit.circuit.library import RYGate
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import Operator

from functions import qft, qft_dagger, controlled_U, controlled_R

dt = np.pi / 8    # time step in e^iHdt
C = 0.01          # constant in controlled-R
n_qubits = 8      
A = np.matrix([
    [1,  0],
    [0, -1]
])

qc = QuantumCircuit(1 + n_qubits + 1, 2)

# |b>
qc.ry(2 * np.arcsin(0.6), n_qubits + 1)

# QPE
qc.barrier(range(1 + n_qubits + 1))
qc.h(range(1, n_qubits + 1))
qc.append(controlled_U(n_qubits, A, dt), range(1, 1 + n_qubits + 1))
qc.append(qft_dagger(n_qubits),range(1, n_qubits + 1))
qc.barrier(range(1 + n_qubits + 1))

# Controlled-R
qc.append(controlled_R(n_qubits, C, dt), range(n_qubits + 1))

# INVERSE QPE
qc.barrier(range(1 + n_qubits + 1))
qc.append(qft(n_qubits),range(1, n_qubits + 1))
qc.append(controlled_U(n_qubits, A, dt).inverse(), range(1, 1 + n_qubits + 1))
qc.h(range(1, n_qubits + 1))
qc.barrier(range(1 + n_qubits + 1))

# measurements
qc.measure(0, 0)
qc.measure(1 + n_qubits, 1)

qc.draw(output='mpl', filename='HHL.png')

# results
backend = BasicSimulator()
t_qc = transpile(qc, backend)
result = backend.run(t_qc, shots=2 ** 21).result()
counts = result.get_counts(t_qc)
print(counts)
Prob_01 = counts['01'] / (counts['11'] + counts['01'])
Prob_11 = counts['11'] / (counts['11'] + counts['01'])
amplitude_01 = np.sqrt(Prob_01)
amplitude_11 = np.sqrt(Prob_11)

print("Prob_01:", Prob_01)
print("Prob_11:", Prob_11)
print(f"x_1: {amplitude_01}")
print(f"x_2: {amplitude_11}")