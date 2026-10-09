"""Run the frozen angle-kernel batch on a quantum backend (fake or real).

Reads data/hw_batch.json, computes the test x support-vector overlap kernel on
the chosen backend, reconstructs the SVC decision, and reports hardware PR-AUC vs
the exact simulator PR-AUC plus the kernel drift (the degradation report).

Backends: apollo / aphrodite = local Q20 / Q50 noise models (NO quota);
          garnet = REAL IQM Garnet via Resonance (SPENDS quota);
          q20 / q50 = real Aalto Q20 / VTT Q50 via LUMI (SPENDS quota).

Run in the IQM venv:
  dry-run:  .venv-iqm/Scripts/python.exe hardware_batch.py --backend apollo
  REAL:     .venv-iqm/Scripts/python.exe hardware_batch.py --backend garnet [--timeslot]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import z_feature_map
from qiskit.quantum_info import Statevector

BATCH = "data/hw_batch.json"


def load_token():
    env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    for line in open(env, encoding="utf-8"):
        if line.strip().startswith("IQM_RESONANCE_TOKEN="):
            t = line.strip().split("=", 1)[1].strip()
            if t and t != "your_iqm_token_here":
                return t
    sys.exit("IQM_RESONANCE_TOKEN not set in .env")


def get_backend(name):
    if name == "apollo":
        from iqm.qiskit_iqm import IQMFakeApollo
        return IQMFakeApollo(), False
    if name == "aphrodite":
        from iqm.qiskit_iqm import IQMFakeAphrodite
        return IQMFakeAphrodite(), False
    if name == "garnet":
        from iqm.qiskit_iqm import IQMProvider
        be = IQMProvider("https://resonance.iqm.tech/", quantum_computer="garnet",
                         token=load_token()).get_backend()
        return be, True
    if name == "q20":
        # Aalto Q20 via LUMI. The fiqci module sets Q20_CORTEX_URL; no Resonance token.
        from iqm.qiskit_iqm import IQMProvider
        be = IQMProvider(os.environ["Q20_CORTEX_URL"], quantum_computer="radiance20").get_backend()
        return be, True
    if name == "q50":
        # VTT Q50 via LUMI. The hackathon module sets Q50_CORTEX_URL and IQM_TOKEN.
        # Pass the URL only: the client reads IQM_TOKEN from the environment itself.
        from iqm.qiskit_iqm import IQMProvider
        be = IQMProvider(os.environ["Q50_CORTEX_URL"], quantum_computer="q50").get_backend()
        return be, True
    sys.exit(f"unknown backend: {name}")


def average_precision(y, score):
    """Average precision (PR-AUC) without sklearn (not in the IQM venv)."""
    order = np.argsort(score)[::-1]
    y = np.asarray(y)[order].astype(float)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(tp[-1], 1)
    ap, prev_r = 0.0, 0.0
    for p, r in zip(precision, recall):
        ap += (r - prev_r) * p
        prev_r = r
    return float(ap)


def q50_counts(job_id, d, shots):
    """Read a finished Q50 job from the QX API.

    job.result() asks for /artifacts/measurements, which this server answers
    with a bad request even after the job status is completed. The counts live
    on GET /api/jobs/{id}. Returns None while the job is still running.
    """
    import requests
    base = os.environ["Q50_CORTEX_URL"].rstrip("/")
    response = requests.get(
        f"{base}/api/jobs/{job_id}",
        headers={
            "Authorization": "Bearer " + os.environ["IQM_TOKEN"],
            "Accept": "application/json",
        },
        timeout=120,
    )
    # This server answers 400 while the job record is not readable yet.
    if response.status_code in (400, 404):
        snippet = " ".join(response.text.split())[:240]
        print(f"  q50 poll {response.status_code}: {snippet}", flush=True)
        return None
    response.raise_for_status()
    results = response.json()["results"]
    status = results.get("status")
    if status in ("failed", "aborted", "error"):
        raise RuntimeError(f"q50 job {job_id} status={status} message={results.get('message')}")
    measurements = results.get("measurements")
    if status != "completed" or not measurements:
        return None
    zeros = "0" * d
    counts = []
    for circuit in measurements:
        columns = [circuit[key] for key in sorted(circuit)]
        n0 = sum(
            1 for shot in range(shots)
            if all(int(columns[q][shot][0]) == 0 for q in range(d))
        )
        counts.append({zeros: n0})
    return counts


def overlap(fmap, a, b, d):
    qc = QuantumCircuit(d)
    qc.compose(fmap.assign_parameters(list(a)), inplace=True)
    qc.compose(fmap.assign_parameters(list(b)).inverse(), inplace=True)
    qc.measure_all()
    return qc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True)
    p.add_argument("--batch", default=BATCH, help="path to the frozen batch json")
    p.add_argument("--shots", type=int, default=1024)
    p.add_argument("--timeslot", action="store_true", help="garnet: use a booked timeslot")
    p.add_argument("--max-per-job", type=int, default=100, help="IQM caps a job at 100 circuits")
    args = p.parse_args()

    B = json.load(open(args.batch))
    d = B["d"]
    sv = np.array(B["sv_angles"]); test = np.array(B["test_angles"])
    dual = np.array(B["dual"]); intercept = B["intercept"]
    ylab = np.array(B["test_labels"]); exact_prauc = B["exact_prauc"]
    fmap = z_feature_map(d, reps=1)
    n_circ = len(test) * len(sv)

    backend, is_real = get_backend(args.backend)
    print(f"backend={backend.name} qubits={backend.num_qubits} | "
          f"circuits={n_circ} x {args.shots} shots | real={is_real}")
    if is_real:
        print(">>> REAL DEVICE — this SPENDS quota. <<<")

    circuits = [overlap(fmap, test[i], sv[j], d)
                for i in range(len(test)) for j in range(len(sv))]
    t0 = time.time()
    tqc = transpile(circuits, backend, optimization_level=3)
    run_kwargs = {"shots": args.shots}
    if args.backend == "garnet":
        run_kwargs["use_timeslot"] = args.timeslot

    counts = []
    n_jobs = (n_circ + args.max_per_job - 1) // args.max_per_job
    for ji, start in enumerate(range(0, n_circ, args.max_per_job), 1):
        chunk = tqc[start:start + args.max_per_job]
        print(f"  job {ji}/{n_jobs} ({len(chunk)} circuits)…", flush=True)
        job = backend.run(chunk, **run_kwargs)
        jid = job.job_id() if hasattr(job, "job_id") else "?"
        print(f"  submitted qpu job id={jid}", flush=True)
        if args.backend == "q50":
            cc = None
            for _ in range(120):
                cc = q50_counts(jid, d, args.shots)
                if cc is not None:
                    break
                print("  waiting for Q50 counts…", flush=True)
                time.sleep(10)
            if cc is None:
                sys.exit(f"q50 job {jid} did not finish within 20 minutes")
        else:
            cc = job.result().get_counts()
            cc = cc if isinstance(cc, list) else [cc]
        counts.extend(cc)
    dt = time.time() - t0

    if len(counts) != n_circ:
        sys.exit(
            f"\nOnly {len(counts)}/{n_circ} circuits returned — some jobs were declined.\n"
            "If you saw 'Not enough credits': on-demand credits aren't available on this\n"
            "account. Hackathon access is via BOOKED TIMESLOTS, so book a Garnet timeslot\n"
            "in IQM Resonance and re-run with --timeslot.")

    K = np.array([counts[k].get("0" * d, 0) / args.shots for k in range(n_circ)]
                 ).reshape(len(test), len(sv))
    dec = K @ dual + intercept
    hw_prauc = average_precision(ylab, dec)

    sv_states = [Statevector(fmap.assign_parameters(list(v))) for v in sv]
    te_states = [Statevector(fmap.assign_parameters(list(v))) for v in test]
    Kx = np.array([[abs(te_states[i].inner(sv_states[j])) ** 2 for j in range(len(sv))]
                   for i in range(len(test))])
    drift = float(np.mean(np.abs(K - Kx)))

    print(f"\nran in {dt:.1f}s")
    print(f"PR-AUC  simulator {exact_prauc:.3f}  ->  {args.backend} {hw_prauc:.3f}  "
          f"(drop {exact_prauc - hw_prauc:+.3f})")
    print(f"kernel drift (mean |hw - exact|): {drift:.3f}")


if __name__ == "__main__":
    main()
