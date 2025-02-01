from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit_aer import QasmSimulator
from matplotlib import pyplot as plt


qr = QuantumRegister(4, 'qr')
cr = ClassicalRegister(4, 'cr')
qc = QuantumCircuit(qr, cr)

qc.h(qr[0])
qc.h(qr[1])
qc.h(qr[2])
qc.h(qr[3])
qc.measure(qr, cr)


backend = QasmSimulator()
result = backend.run(qc.reverse_bits(), shots=1e4).result()
counts = result.get_counts()


plot_histogram(counts, title="Quantum Random Number Generator (4 Qubits)")
plt.show()