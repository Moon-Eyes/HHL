import numpy as np
from qiskit import QuantumCircuit, transpile
import torch
from qiskit.circuit.library import RYGate
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import Operator


def qft(n_qubits):
    circuit = QuantumCircuit(n_qubits)
    circuit.h(range(n_qubits))                # 初始化钟表的指针
    for i in reversed(range(n_qubits)):
        circuit.barrier(range(n_qubits))
        for j in reversed(range(i)):
            circuit.crx(np.pi / 2 ** (i - j),
                        j,                    # 控制位bit
                        i)                    # 目标位bit
    return circuit


def qft_dagger(n_qubits):
    return qft(n_qubits).inverse()


def controlled_U(n_qubits, A, dt):
    circuit = QuantumCircuit(n_qubits + 1)
    U = np.matrix(torch.matrix_exp(torch.tensor(1j * A * dt)))  
    for i in range(n_qubits):
        
        sub_circuit = QuantumCircuit(1)  
        sub_circuit.name = '$U^{2^' + str(i) + '}$'
        sub_circuit.append(Operator(U ** (2 ** i)), [0])         
        sub_gate = sub_circuit.to_gate().control()             
        circuit.append(sub_gate, [n_qubits - i - 1, n_qubits])
    return circuit


def controlled_R(n_qubits, C, dt):
    circuit = QuantumCircuit(n_qubits + 1)
    for k in range(1, 2 ** n_qubits):
        for i in range(n_qubits):  
            if (k & (1 << i)) == 0:
                circuit.x(1 + i)
        phi = k / (2 ** n_qubits)
        if k > 2 ** (n_qubits - 1):
            phi -= 1.0  
        l = 2 * np.pi / dt * phi
        circuit.append(RYGate(2 * np.arcsin(C / l)).control(n_qubits),
                       [(i + 1) % (n_qubits + 1) for i in range(n_qubits + 1)])

        for i in range(n_qubits): 
            if (k & (1 << i)) == 0:
                circuit.x(1 + i)
        circuit.barrier(range(n_qubits + 1))
    return circuit
