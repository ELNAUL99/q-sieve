# Q-SIEVE — Quantum workstream guide

How to run the quantum part: see it work, compare encodings, and run on real
hardware (IQM Garnet + IBM). Three isolated venvs because the hardware libraries
conflict with the ML stack's qiskit:
- `.venv` — ML stack (qiskit, scikit-learn, xgboost)
- `.venv-iqm` — `iqm-client[qiskit]`
- `.venv-ibm` — `qiskit-ibm-runtime`

## 1. See the kernel run
```powershell
.\.venv\Scripts\python.exe quantum_demo.py
```
Encode data into qubits → fidelity similarity matrix → shots-based estimate (how
hardware does it).

## 2. Compare encodings (seed-repeated, noise-aware)
```powershell
.\.venv\Scripts\python.exe sweep.py
```
Finding: amplitude/angle > ZZ; kernel training didn't help; all near the floor.

## 3. Choose the hardware config
Angle encoding — shallow, zero two-qubit gates, most NISQ-robust (noise sweep:
angle drift 0.010 vs amplitude 0.158). Keep amplitude as a simulator-only ceiling.

## 4. Real hardware

### IQM Garnet (20-qubit, Resonance)
Confirmed: server `https://resonance.iqm.tech/`, alias `garnet`, token
`IQM_RESONANCE_TOKEN` in `.env`.

```powershell
.\.venv-iqm\Scripts\python.exe garnet_connect.py                    # no-cost check
.\.venv-iqm\Scripts\python.exe hardware_batch.py --backend apollo   # local dry-run (no quota)
.\.venv-iqm\Scripts\python.exe hardware_batch.py --backend garnet   # REAL (spends credits)
```
IQM caps a job at 100 circuits (the runner chunks automatically). On-demand uses
credits; a booked timeslot uses `--timeslot`.

### IBM (156-qubit Heron)
Needs `IBM_QUANTUM_TOKEN` + `IBM_QUANTUM_CRN` in `.env`.
```powershell
.\.venv-ibm\Scripts\python.exe ibm_connect.py    # no-cost check, lists backends + max_circuits
.\.venv-ibm\Scripts\python.exe ibm_batch.py      # REAL (IBM's free allocation, not IQM credits)
```

## 5. The degradation report
For the frozen config, report simulator PR-AUC vs hardware PR-AUC + kernel drift.

**Result (three real QPUs, three vendors):**
| Device | PR-AUC sim → hw | drop | drift |
|---|---|---|---|
| IBM marrakesh (156q) | 0.111 → 0.111 | 0.000 | 0.007 |
| IQM Garnet (20q) | 0.111 → 0.111 | 0.000 | 0.010 |
| VTT Q50 (50q) | 0.111 → 0.100 | +0.011 | 0.018 |

Two vendors show zero degradation; the noisiest chip (VTT Q50, drift 0.018 — 2–3×
the others) costs just one PR-AUC point and still clears the ~0.093 floor. Angle
encoding (depth 2, zero two-qubit gates) is shallow enough that even Q50's noise
barely moves it. The hardware batch uses 6 features (6 qubits) to keep readout
noise low — the full 12-feature model is for the simulator benchmark.

## Pipeline to produce the hardware batch
```powershell
.\.venv\Scripts\python.exe export_hw_batch.py    # freeze -> data/hw_batch.json
```
Then run it on any backend with `hardware_batch.py` / `ibm_batch.py`.
