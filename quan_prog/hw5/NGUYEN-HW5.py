from qiskit import *
from qiskit.circuit.library import UnitaryGate, QFT
from qiskit_aer import QasmSimulator
from math import gcd, floor, log
from fractions import Fraction
import numpy as np


def mod_mult_gate(a, N):
    if gcd(a, N) != 1:
        raise ValueError(f"gcd({a}, {N}) ≠ 1")

    n = int(np.ceil(np.log2(N)))
    dim = 2**n
    U = np.zeros((dim, dim))

    for x in range(dim):
        if x < N:
            U[a * x % N, x] = 1
        else:
            U[x, x] = 1

    U_gate = UnitaryGate(U, label=f"U_{a}")
    return U_gate

# pow(3, 2, 4)



def period_finding_circuit(a, N):
    if gcd(a, N) != 1:
        raise ValueError(f"gcd({a}, {N}) ≠ 1")

    n = int(np.ceil(np.log2(N)))
    m = 2 * n  # often works well

    qc = QuantumCircuit(m + n, m)
    qc.h(range(m))  # Apply Hadamards to top register
    qc.x(m + n -1)  # Initialize bottom register to |0000...01⟩

    for i in range(m):
        power = pow(a, 2**i, N)
        control_qubit = m - 1 - i
        controlled_U = mod_mult_gate(power, N).control()
        qc.compose(controlled_U, qubits=[control_qubit] + list(range(m, m + n)), inplace=True)


    # qc.append(QFT(m, do_swaps=False).inverse(), range(m))
    qc.append(QFT(m, do_swaps=False), range(m))

    qc.barrier()
    qc.measure(range(m), range(m))

    return qc


def run_QPF(a, N):
    if gcd(a, N) != 1:
        raise ValueError(f"gcd({a}, {N}) ≠ 1")

    qc = period_finding_circuit(a, N)
    backend = QasmSimulator()
    transpiled_circuit = transpile(qc, backend)
    result = backend.run(transpiled_circuit.reverse_bits()).result() # run once
    counts = result.get_counts()
    measured = max(counts, key=counts.get)  # most likely result

    y = int(measured, 2)
    return y


def closest_fraction(y, K, N):
    return Fraction(y, K).limit_denominator(N).denominator


def find_order(a, N, max_attempts=100):
    if gcd(a, N) != 1:
        raise ValueError(f"gcd({a}, {N}) ≠ 1")

    n = int(np.ceil(np.log2(N)))
    m = 2 * n
    K = 2**m

    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        y = run_QPF(a, N)
        r_candidate = closest_fraction(y, K, N)

        if r_candidate == 0:
            continue  # skip invalid values

        # print(f"  QPF suggested r = {r_candidate}")

        for k in range(1, N):
            r = k * r_candidate
            if pow(a, r, N) == 1:
                return r, attempts

    raise RuntimeError("Failed to find order within max_attempts")



if __name__ == "__main__":
    try:
        a = int(input("Enter a: "))
        N = int(input("Enter N: "))
        r, loops = find_order(a, N)
        print(f"Order of {a} mod {N} is {r}, found in {loops} iteration(s).")
    except Exception as e:
        print(f"Error: {e}")

    # Example usage:
    # Enter a: 2
    # Enter N: 5
    # Order of 2 mod 5 is 4, found in 2 iteration(s).
