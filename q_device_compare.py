"""Q20 vs Q50 noise comparison (IQMFakeApollo vs IQMFakeAphrodite), local.

Runs the SAME circuits (angle + ZZ-linear) on both device noise models and
compares drift + readout floor.

Run:  .venv-iqm/Scripts/python.exe q_device_compare.py
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import z_feature_map, zz_feature_map
from qiskit.quantum_info import Statevector
from iqm.qiskit_iqm import IQMFakeApollo, IQMFakeAphrodite

D = 6
N = 6
SHOTS = 2048
SEED = 0
rng = np.random.default_rng(SEED)
VECS = rng.uniform(0, np.pi, size=(N, D))

DEVICES = [("Q20/Apollo", IQMFakeApollo()), ("Q50/Aphrodite", IQMFakeAphrodite())]
ENCODINGS = {
    "angle r1": z_feature_map(D, reps=1),
    "zz-lin r1": zz_feature_map(D, reps=1, entanglement="linear"),
}


def overlap(fmap, x, z):
    qc = QuantumCircuit(D)
    qc.compose(fmap.assign_parameters(list(x)), inplace=True)
    qc.compose(fmap.assign_parameters(list(z)).inverse(), inplace=True)
    qc.measure_all()
    return qc


def measure(backend, fmap):
    states = [Statevector(fmap.assign_parameters(list(v))) for v in VECS]
    drifts, self_ov, depths, twoq = [], [], [], []
    for i in range(N):
        for j in range(i, N):
            exact = float(abs(states[i].inner(states[j])) ** 2)
            tqc = transpile(overlap(fmap, VECS[i], VECS[j]), backend, optimization_level=3, seed_transpiler=SEED)
            p0 = backend.run(tqc, shots=SHOTS).result().get_counts().get("0" * D, 0) / SHOTS
            if i == j:
                self_ov.append(p0)
            else:
                drifts.append(abs(p0 - exact)); depths.append(tqc.depth())
                twoq.append(sum(1 for d in tqc.data if d.operation.num_qubits == 2))
    return (float(np.mean(drifts)), float(1 - np.mean(self_ov)),
            float(np.mean(depths)), float(np.mean(twoq)))


def main():
    print(f"sample N={N}, qubits={D}, shots={SHOTS}\n")
    rows = []
    for enc, fmap in ENCODINGS.items():
        for dname, be in DEVICES:
            drift, self_drop, depth, tq = measure(be, fmap)
            rows.append((enc, dname, be.num_qubits, depth, tq, drift, self_drop))
            print(f"  {enc:<10} {dname:<14} qubits={be.num_qubits:>2} "
                  f"depth={depth:5.1f} 2q={tq:4.1f} drift={drift:.3f} self_drop={self_drop:.3f}")
    print("\nQ20 vs Q50 per encoding (positive Q50-Q20 = Q50 noisier)")
    for enc in ENCODINGS:
        q20 = next(r for r in rows if r[0] == enc and "Q20" in r[1])
        q50 = next(r for r in rows if r[0] == enc and "Q50" in r[1])
        print(f"  {enc:<10} drift Q20={q20[5]:.3f} Q50={q50[5]:.3f} diff={q50[5]-q20[5]:+.3f}")


if __name__ == "__main__":
    main()
