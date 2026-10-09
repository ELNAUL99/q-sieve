"""Noise comparison of every encoding on the Aalto Q20 model (IQM Apollo).

Reports per encoding: drift (mean |Q20 - exact| off-diagonal), self_drop (readout
floor), transpiled depth and 2-qubit-gate count.

Run:  .venv-iqm/Scripts/python.exe q20_noise_sweep.py
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import z_feature_map, zz_feature_map, StatePreparation
from qiskit.quantum_info import Statevector
from iqm.qiskit_iqm import IQMFakeApollo

D = 6
N = 6
SHOTS = 2048
SEED = 0
rng = np.random.default_rng(SEED)
ANGLES = rng.uniform(0, np.pi, size=(N, D))
AMP = rng.uniform(0, 1, size=(N, 8)); AMP /= np.linalg.norm(AMP, axis=1, keepdims=True)
backend = IQMFakeApollo()


def param_map(kind):
    if kind == "angle r1":   return z_feature_map(D, reps=1)
    if kind == "angle r2":   return z_feature_map(D, reps=2)
    if kind == "zz-lin r1":  return zz_feature_map(D, reps=1, entanglement="linear")
    if kind == "zz-lin r2":  return zz_feature_map(D, reps=2, entanglement="linear")
    if kind == "zz-full r1": return zz_feature_map(D, reps=1, entanglement="full")
    if kind == "zz trained-struct":
        from qiskit.circuit import ParameterVector
        th = ParameterVector("t", D)
        fm0 = QuantumCircuit(D)
        for i in range(D):
            fm0.ry(th[i], i)
        fm = fm0.compose(zz_feature_map(D, reps=1, entanglement="linear"))
        return fm.assign_parameters({th[i]: 0.5 for i in range(D)})
    raise ValueError(kind)


def encode_param(kind, vec):
    return param_map(kind).assign_parameters(list(vec))


def encode_amp(vec):
    qc = QuantumCircuit(3)
    qc.append(StatePreparation(vec), range(3))
    return qc


def overlap(circ_x, circ_z):
    qc = QuantumCircuit(circ_x.num_qubits)
    qc.compose(circ_x, inplace=True)
    qc.compose(circ_z.inverse(), inplace=True)
    qc.measure_all()
    return qc


def run_encoding(kind):
    is_amp = kind == "amplitude"
    data = AMP if is_amp else ANGLES
    enc = (lambda v: encode_amp(v)) if is_amp else (lambda v: encode_param(kind, v))
    circuits = [enc(v) for v in data]
    states = [Statevector(c) for c in circuits]
    depths, twoq, drifts, self_overlaps = [], [], [], []
    for i in range(N):
        for j in range(i, N):
            exact = float(abs(states[i].inner(states[j])) ** 2)
            tqc = transpile(overlap(circuits[i], circuits[j]), backend, optimization_level=3, seed_transpiler=SEED)
            nq = circuits[i].num_qubits
            p0 = tqc and backend.run(tqc, shots=SHOTS).result().get_counts().get("0" * nq, 0) / SHOTS
            if i == j:
                self_overlaps.append(p0)
            else:
                drifts.append(abs(p0 - exact)); depths.append(tqc.depth())
                twoq.append(sum(1 for inst in tqc.data if inst.operation.num_qubits == 2))
    return {"encoding": kind, "qubits": circuits[0].num_qubits,
            "depth": float(np.mean(depths)), "2q_gates": float(np.mean(twoq)),
            "drift": float(np.mean(drifts)), "self_drop": float(1 - np.mean(self_overlaps))}


def main():
    print(f"backend: {backend.name} | sample N={N} | shots={SHOTS}\n")
    order = ["angle r1", "angle r2", "zz-lin r1", "zz-lin r2",
             "zz trained-struct", "zz-full r1", "amplitude"]
    rows = []
    for kind in order:
        r = run_encoding(kind)
        rows.append(r)
        print(f"  {r['encoding']:<18} qubits={r['qubits']} depth={r['depth']:5.1f} "
              f"2q={r['2q_gates']:5.1f} drift={r['drift']:.3f} self_drop={r['self_drop']:.3f}")
    rows.sort(key=lambda r: r["drift"])
    print("\nRANKED BY NOISE ROBUSTNESS (lowest drift first)")
    for r in rows:
        print(f"  {r['encoding']:<18} drift={r['drift']:.3f} self_drop={r['self_drop']:.3f}")


if __name__ == "__main__":
    main()
