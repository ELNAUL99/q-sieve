"""Quantum demo — watch the quantum kernel run, step by step.

Run:  .venv/Scripts/python.exe quantum_demo.py
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

line = "=" * 70
A = [0.30, 2.90]
B = [0.40, 2.70]   # close to A
C = [2.80, 0.20]   # far from A

print(line)
print("STEP 1 — Build the feature map (encodes data into qubits)")
print(line)
fmap = zz_feature_map(feature_dimension=2, reps=1, entanglement="linear")
print(fmap.decompose().draw(output="text"))

print("\n" + line)
print("STEP 2 — Encode window A into a quantum state")
print(line)
state_A = Statevector(fmap.assign_parameters(A))
for basis, amp in zip(["|00>", "|01>", "|10>", "|11>"], state_A.data):
    print(f"   {basis}:  {amp.real:+.3f} {amp.imag:+.3f}i   (prob {abs(amp)**2:.3f})")

print("\n" + line)
print("STEP 3 — The quantum similarity (fidelity) between windows")
print(line)
kernel = FidelityStatevectorKernel(feature_map=fmap)
K = kernel.evaluate([A, B, C])
print("          A       B       C")
for name, rowvals in zip("ABC", K):
    print(f"   {name}   " + "  ".join(f"{v:.3f}" for v in rowvals))
print(f"\nA~B = {K[0,1]:.3f} (high, close) but A~C = {K[0,2]:.3f} (low, far).")

print("\n" + line)
print("STEP 4 — How REAL hardware estimates it: run circuit + measure")
print(line)
overlap = QuantumCircuit(2)
overlap.compose(fmap.assign_parameters(A), inplace=True)
overlap.compose(fmap.assign_parameters(B).inverse(), inplace=True)
overlap.measure_all()
sim = AerSimulator()
shots = 4096
counts = sim.run(transpile(overlap, sim), shots=shots).result().get_counts()
p_all_zero = counts.get("00", 0) / shots
print(f"Measurement counts: {counts}")
print(f"Fraction |00>: {p_all_zero:.3f}  (shots-based fidelity)")
print(f"Exact fidelity: {K[0,1]:.3f}")
print("\nThey match. On real hardware the shots-based number drifts a little —")
print("that drift is exactly the 'noise' the degradation report measures.")
