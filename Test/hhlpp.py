import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RYGate
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import Operator
import matplotlib.pyplot as plt
import time

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
    U = np.matrix(expm(1j * A * dt)) 
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


if __name__ == "__main__":
    import scipy.linalg


    dt = np.pi / 8    # time step in e^iHdt
    C = 0.01          # constant in controlled-R
    n_qubits = 2      
    A = np.matrix([
        [1,  0],
        [0, -1]
    ])


    qc = QuantumCircuit(1 + n_qubits + 1, 2)
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

    
    decomposed_qc = qc.decompose()
    fig = decomposed_qc.draw('mpl', fold=-1, scale=0.8)
    plt.title(f"HHL Quantum Circuit ({n_qubits} Clock Qubits)", fontsize=14)
    plt.savefig('HHL_detailed.png', bbox_inches='tight', dpi=300)
    print("电路图已保存为 HHL_detailed.png")
    
    # results
    print("运行量子模拟:")
    start_time = time.time()
    backend = BasicSimulator()
    t_qc = transpile(qc, backend)
    result = backend.run(t_qc, shots=2 ** 21).result()

    end_time = time.time()  
    elapsed_time = end_time - start_time 
    counts = result.get_counts(t_qc)
    print(counts)
    count_01 = counts.get('01', 0)
    count_11 = counts.get('11', 0)
    total = count_01 + count_11

if total > 0:
    Prob_01 = count_01 / total
    Prob_11 = count_11 / total
    amplitude_01 = np.sqrt(Prob_01)
    amplitude_11 = np.sqrt(Prob_11)
    print("Prob_01:", Prob_01)
    print("Prob_11:", Prob_11)
    print(f"x=({amplitude_01}, {amplitude_11})")
else:
    print("未测量到有效解!")

print(f"运行时间: {elapsed_time:.2f} 秒")
    