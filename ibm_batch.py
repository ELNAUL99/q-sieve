"""Run the frozen angle-kernel batch on a real IBM backend (SamplerV2).

Uses IBM's own free allocation (not IQM credits). Reuses data/hw_batch.json and
the analysis helpers from hardware_batch.py, so the result is directly
comparable to the Garnet run.

Run in the IBM venv:
    .venv-ibm/Scripts/python.exe ibm_batch.py                 # least-busy backend
    .venv-ibm/Scripts/python.exe ibm_batch.py --backend ibm_fez --shots 512
"""

import argparse
import json
import time

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import z_feature_map
from qiskit.quantum_info import Statevector

from hardware_batch import average_precision, overlap, BATCH
from ibm_connect import env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default=None, help="backend name; default = least busy")
    p.add_argument("--shots", type=int, default=1024)
    args = p.parse_args()

    B = json.load(open(BATCH))
    d = B["d"]
    sv = np.array(B["sv_angles"]); test = np.array(B["test_angles"])
    dual = np.array(B["dual"]); intercept = B["intercept"]
    ylab = np.array(B["test_labels"]); exact_prauc = B["exact_prauc"]
    fmap = z_feature_map(d, reps=1)
    n_circ = len(test) * len(sv)

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    service = QiskitRuntimeService(channel="ibm_quantum_platform",
                                   token=env("IBM_QUANTUM_TOKEN"),
                                   instance=env("IBM_QUANTUM_CRN"))
    backend = service.backend(args.backend) if args.backend else \
        service.least_busy(operational=True, simulator=False)
    print(f">>> REAL IBM backend={backend.name} qubits={backend.num_qubits} | "
          f"circuits={n_circ} x {args.shots} shots — uses IBM allocation <<<")

    circuits = [overlap(fmap, test[i], sv[j], d)
                for i in range(len(test)) for j in range(len(sv))]
    tqc = transpile(circuits, backend, optimization_level=1)
    creg = tqc[0].cregs[0].name

    # Use IBM's OWN reported per-job limit (not a guess); 1 job if it fits.
    per_job = getattr(backend, "max_circuits", None) or n_circ
    sampler = SamplerV2(mode=backend)
    counts = []
    t0 = time.time()
    n_jobs = (n_circ + per_job - 1) // per_job
    print(f"  backend max_circuits={per_job} -> {n_jobs} job(s)")
    for ji, start in enumerate(range(0, n_circ, per_job), 1):
        chunk = tqc[start:start + per_job]
        print(f"  job {ji}/{n_jobs} ({len(chunk)} circuits)…", flush=True)
        res = sampler.run(chunk, shots=args.shots).result()
        counts.extend(getattr(r.data, creg).get_counts() for r in res)
    dt = time.time() - t0

    K = np.array([c.get("0" * d, 0) / args.shots for c in counts]).reshape(len(test), len(sv))
    dec = K @ dual + intercept
    hw_prauc = average_precision(ylab, dec)

    sv_states = [Statevector(fmap.assign_parameters(list(v))) for v in sv]
    te_states = [Statevector(fmap.assign_parameters(list(v))) for v in test]
    Kx = np.array([[abs(te_states[i].inner(sv_states[j])) ** 2 for j in range(len(sv))]
                   for i in range(len(test))])
    drift = float(np.mean(np.abs(K - Kx)))

    print(f"\nran in {dt:.1f}s on {backend.name}")
    print(f"PR-AUC  simulator {exact_prauc:.3f}  ->  {backend.name} {hw_prauc:.3f}  "
          f"(drop {exact_prauc - hw_prauc:+.3f})")
    print(f"kernel drift (mean |hw - exact|): {drift:.3f}")


if __name__ == "__main__":
    main()
