"""Run the Q-SIEVE kernel on a LOCAL simulation of Aalto Q20 (IQM Apollo).

IQMFakeApollo is the 20-qubit Apollo noise model — real Q20 connectivity + noise,
on your laptop, no tokens, no queue. Angle encoding (shallow, NISQ-robust).

Run:  .venv-iqm/Scripts/python.exe q20_local_demo.py
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import z_feature_map
from qiskit.quantum_info import Statevector
from iqm.qiskit_iqm import IQMFakeApollo

N = 4
A = [0.30, 2.90, 1.10, 0.40]
B = [0.40, 2.70, 1.00, 0.50]
C = [2.80, 0.20, 0.30, 2.60]
fmap = z_feature_map(feature_dimension=N, reps=1)


def exact_fidelity(x, z):
    return float(abs(Statevector(fmap.assign_parameters(x)).inner(
        Statevector(fmap.assign_parameters(z)))) ** 2)


def overlap_circuit(x, z):
    qc = QuantumCircuit(N)
    qc.compose(fmap.assign_parameters(x), inplace=True)
    qc.compose(fmap.assign_parameters(z).inverse(), inplace=True)
    qc.measure_all()
    return qc


def main():
    backend = IQMFakeApollo()
    print(f"local backend: {backend.name} | qubits: {backend.num_qubits}")
    print("encoding: angle (z_feature_map, reps=1) — NISQ-safe\n")
    shots = 4096
    print(f"{'pair':<6} {'exact':>7} {'Q20-noise':>10} {'drift':>7} {'depth':>6}")
    for (nx, x), (nz, z) in [(("A", A), ("B", B)), (("A", A), ("C", C))]:
        exact = exact_fidelity(x, z)
        tqc = transpile(overlap_circuit(x, z), backend, optimization_level=3)
        counts = backend.run(tqc, shots=shots).result().get_counts()
        p0 = counts.get("0" * N, 0) / shots
        print(f"{nx}~{nz:<4} {exact:>7.3f} {p0:>10.3f} {abs(p0-exact):>7.3f} {tqc.depth():>6}")
    print("\nThe drift is what the hardware degradation report measures.")


if __name__ == "__main__":
    main()
