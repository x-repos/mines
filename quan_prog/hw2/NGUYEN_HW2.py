from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import QasmSimulator
from qiskit.visualization import plot_histogram

def state_preparation():
    """Prepares the initial Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2."""
    qr = QuantumRegister(2, 'qr')
    cr = ClassicalRegister(2, 'cr')
    qc = QuantumCircuit(qr, cr)
    
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])
    
    return qc

def encode(qc, x):
    """Encodes a 2-bit message x onto the quantum state."""
    if x not in ['00', '01', '10', '11']:
        raise ValueError("Input x must be a 2-bit string: '00', '01', '10', or '11'")
    
    if x == '00':
        qc.id(0)  # Identity gate (no change)
    elif x == '01':
        qc.x(0)  # Bit-flip
    elif x == '10':
        qc.z(0)  # Phase-flip
    elif x == '11':
        qc.y(0)  # Bit & Phase flip
    
    return qc

def decode(qc):
    """Applies the decoding operations for Bob to recover x."""
    qc.cx(0, 1)
    qc.h(0)
    return qc

def measure(qc):
    """Adds measurement gates to the circuit."""
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc

def run_protocol(qc):
    """Runs the quantum protocol on a QASM simulator."""
    backend = QasmSimulator()
    result = backend.run(qc.reverse_bits(), shots=int(1e4)).result()
    counts = result.get_counts()
    return counts

def main():
    """Executes the quantum communication protocol."""
    qc = state_preparation()
    qc.barrier()
    
    x = input("Message (00, 01, 10, 11): ").strip()
    
    qc = encode(qc, x)
    qc.barrier()
    
    qc = decode(qc)
    qc.barrier()
    
    qc = measure(qc)
    
    print(run_protocol(qc))
    print(qc)

if __name__ == "__main__":
    main()
