# Q-SIEVE — results log (lab notebook)

A dated history of experiments. Each entry records the **exact conditions** it
ran under, the **result**, and **what changed vs the previous run** — so a number
is never ambiguous later. Raw snapshots: `results/` (git-tracked, timestamped).

**How to add an entry:** run the experiment → snapshot its CSV into `results/`
→ add a section below with Conditions / Result / vs last time.

**Feature versions** (numbers compare only within a version):
- **v1** — 11 within-window features, amounts NOT currency-normalized, calendar-week windows.
- **v2** — 12 features: v1 + amounts converted to USD (15-currency FX table) +
  `velocity_ratio` (window tx-count ÷ the account's own baseline).

`data/X.csv` (v2): 44,084 flagged windows, 9.3% suspicious. Random-guess PR-AUC ≈ **0.093**.

---

## 1 · Encoding sweep — 2026-10-07

**Conditions**
- Data: v1 (11 features), real IBM AML HI-Small, 44,084-window pool.
- Train/test: `train_cap=150` (stratified), `test_cap=1500`; seeds = 5.
- Scaler: **MinMax[0,π] fixed** (same for every quantum config).
- Models: SVC `class_weight="balanced"`; quantum via `FidelityStatevectorKernel` (Qiskit QML).
- Compute: **noise-free statevector simulator**. Metric: PR-AUC (mean ± std).

**Result** — `results/20261008_encoding_sweep.csv`

| config | PR-AUC | vs best |
|---|---|---|
| XGBoost (near-default) | 0.134 ± 0.013 | best |
| Amplitude (sim-only) | 0.127 ± 0.019 | ~tie (noise) |
| Angle r1 | 0.119 ± 0.010 | ~tie |
| ZZ linear r1 | 0.112 ± 0.012 | ~tie |
| ZZ linear r2 | 0.108 ± 0.013 | ~tie |
| ZZ full r1 / r2 | 0.106 | worse |
| PCA4→Amp | 0.102 | worse |
| ZZ trained | 0.100 | worse |
| RBF-SVC | 0.097 | worse |
| PCA2→Amp | 0.097 | worse |

**vs last time:** first *disciplined* sweep — added seed-repeats + noise-aware
ranking, and the trained-kernel + amplitude + PCA→amplitude configs. Before this
we only had single-number, single-seed comparisons. **Finding:** amplitude/angle
> ZZ; kernel *training* did NOT help; everything near the floor.

## 2 · Q20 noise sweep — 2026-10-07

**Conditions**
- Backend: **`IQMFakeApollo`** — 20-qubit Apollo *noise model*, run **locally** (not real hardware).
- Circuit: 6 features → 6 qubits; sample = 6 random vectors in [0,π]; shots = 2048.
- NOT on AML data (random vectors isolate *circuit* noise, not data signal).
- Metric: **drift** = mean |noisy − exact| off-diagonal; **self_drop** = 1 − mean self-overlap (noise floor).

**Result**

| encoding | qubits | depth | 2q gates | drift | self_drop |
|---|---|---|---|---|---|
| angle r1 | 6 | 2 | 0 | 0.010 | 0.265 |
| ZZ-lin r1 | 6 | 38 | 18 | 0.009 | 0.260 |
| ZZ-lin r2 | 6 | 54 | 38 | 0.008 | 0.266 |
| ZZ trained-struct | 6 | 38 | 18 | 0.020 | 0.270 |
| ZZ-full r1 | 6 | 138 | 84 | 0.021 | 0.276 |
| **amplitude** | 3 | 22 | 10 | **0.158** | 0.158 |

**vs last time:** first hardware-*noise* experiment (sweep #1 was noise-free).
**Finding:** readout error dominates the floor (~0.26 for 6 qubits, depth-independent);
**amplitude is NOT NISQ-suitable** (15× worse drift); training ≈ fixed (same depth).
→ **angle encoding is the hardware config** (shallow, 0 two-qubit gates).

## 3 · Fair benchmark — 2026-10-08

**Conditions**
- Data: **v2 (12 features)**, HI-Small, 44,084 pool.
- Train/test: `train_cap=200` (identical for ALL models), `test_cap=3000`; seeds = 3.
- Quantum: **angle encoding**, MinMax[0,π], SVC `C` tuned by 3-fold CV.
- Baselines: **GridSearchCV-tuned** XGBoost (depth/lr/n_est/min_child) and RBF (C/gamma), imbalance-aware.
- Metrics: PR-AUC, **FP eliminated @90% recall**, **auto-clear @≤1% error** (uncertain-case split). Simulator.

**Result** — `results/20261008_benchmark.csv`

| model | PR-AUC | FP-elim @90% rec | auto-clear @≤1% |
|---|---|---|---|
| **XGBoost (tuned)** | **0.190 ± 0.005** | **17.8%** | 0% |
| quantum (angle) | 0.125 ± 0.021 | 14.6% | 0% |
| RBF (tuned) | 0.097 ± 0.011 | 11.5% | 0.2% |

**vs last time (vs sweep #1):**
- Features **v1 → v2** (added currency normalization + velocity_ratio).
- Baselines **default → GridSearch-tuned** (the fair-benchmark fix).
- `train_cap` **150 → 200**; quantum config **ZZ → angle** (the NISQ pick).
- **New metrics:** FP-eliminated-at-recall + auto-clear split (false positives + uncertain cases).
- **Effect:** tuning lifted XGBoost **0.134 → 0.190**; the gap over quantum now
  *exceeds* the noise band → **classical clearly wins** (was "~tie" before).
  Quantum(angle) still beats the RBF kernel. No safe auto-clear yet.

## 4 · Scaling sweep — 2026-10-08

**Conditions**
- Data: v2 (12 features); angle kernel; `train_cap=200`, `test_cap=2000`; seeds = 5.
- Transforms: MinMax [0,π]/[0,π/2]/[0,2π], Standard, **Robust, log1p, Quantile, Power(YJ)** → [0,π].
- Metric: PR-AUC (mean ± std), noise-aware ranking. Simulator.

**Result** — `results/20261008_scaling.csv`

| scaling | PR-AUC | note |
|---|---|---|
| MinMax [0,π/2] | 0.138 ± 0.031 | highest mean, but noisiest |
| Power(YJ)→[0,π] | 0.137 ± 0.021 | ~tie |
| log1p→[0,π] | 0.134 ± 0.022 | ~tie |
| **Quantile→[0,π]** | 0.127 ± **0.012** | most *stable* lift |
| MinMax [0,π] (baseline) | 0.118 ± 0.019 | — |
| Standard / Robust →[0,π] | 0.118 ± 0.019 | — |
| MinMax [0,2π] | 0.108 ± 0.008 | worst (angle aliasing) |

**vs last time:** first test of **distribution-shaping** scalers (sweep #1 only
tried linear MinMax/Standard). **Finding:** shapers (Power/log1p/Quantile) do land
above plain MinMax (0.118 → 0.13–0.14) — the heavy-tail hypothesis is directionally
right — **but within the noise band**, so not a provable win. The top-by-mean
(π/2) is the *noisiest* (std 0.031) = a lucky result, not reliable; the trustworthy
signal is **Quantile** (0.127 ± 0.012, tightest bars). `[0,2π]` is worst (aliasing).
Even the best stays well below tuned XGBoost (0.190). → scaling is a real but small
lever; **features remain the ceiling**. Adopt Quantile (stable + tames outliers).

## 5 · REAL HARDWARE — two vendors, zero degradation (2026-10-08) ⭐⭐

**Conditions:** frozen angle-kernel SVC (`export_hw_batch.py`), 6 qubits, 41
support vectors × 16 test = **656 overlap circuits** × 1024 shots. Same batch on
both devices. `results/20261008_hardware_real.csv`.

| backend | vendor | qubits | PR-AUC sim → hw | drop | drift | time |
|---|---|---|---|---|---|---|
| **Garnet (REAL)** | IQM | 20 | 0.111 → **0.111** | **0.000** | **0.010** | 252 s |
| **ibm_marrakesh (REAL)** | IBM | 156 | 0.111 → **0.111** | **0.000** | **0.007** | 613 s |
| Apollo (Q20 model) | IQM-sim | 20 | 0.111 → 0.100 | 0.011 | 0.032 | 12 s |
| Aphrodite (Q50 model) | IQM-sim | 54 | 0.111 → 0.100 | 0.011 | 0.033 | 13 s |

**Finding:** the quantum kernel ran on **two real quantum computers from two
different vendors (IQM + IBM) with ZERO degradation** — PR-AUC identical to
simulation on both, drift 0.007–0.010. This validates the whole encoding
analysis: angle (depth 2, 0 two-qubit gates) is shallow enough that neither
chip's noise touches it. Both **real chips beat the noise models** (0.007–0.010
drift vs 0.032 modeled). Quantum is still < tuned XGBoost (0.190), but the
quantum result is now **hardware-validated on real metal, twice over**. This is
the credibility slide. (Both devices enforce a 100-circuit/job limit → 7 jobs.)

---

## Noise — how each encoding survives hardware

From sweep #2 (Q20/Apollo model). The ranking is driven by circuit depth + 2-qubit
gates; the shared ~0.26 self-drop is 6-qubit readout error.

- **angle** — drift 0.010, depth 2, **0** two-qubit gates → best + fastest. **Hardware pick.**
- **ZZ-linear** — drift ~0.009, 18–38 two-qubit gates → fine on the model, but real
  hardware usually punishes 2q gates more than the model shows.
- **ZZ-full** — depth 138, 84 two-qubit gates (SWAPs on limited connectivity) → avoid.
- **amplitude** — drift **0.158** (15× worse) → the kernel structure is wrecked; sim-only.

**Caveat:** fake models tend to *understate* real gate error, so treat these as
optimistic lower bounds. Angle's zero-2q-gate design hedges that uncertainty.

**Q20 vs Q50 difference** (2026-10-08, `results/20261008_q20_vs_q50.csv`) — same
angle + ZZ-linear circuits, 6 qubits, N=6, shots=2048, on `IQMFakeApollo` (20q)
vs `IQMFakeAphrodite` (54q):

| encoding | metric | Q20 | Q50 | Q50−Q20 |
|---|---|---|---|---|
| angle | drift | 0.012 | 0.009 | −0.002 |
| angle | self_drop | 0.276 | 0.273 | −0.003 |
| ZZ-linear | drift | 0.008 | 0.009 | +0.001 |
| ZZ-linear | self_drop | 0.269 | 0.266 | −0.003 |

**Finding (corrected a prior guess):** I expected Q50 ≥ Q20 drift; the models show
them **indistinguishable** (±0.001–0.003, inside the seed wobble). We only use 6 of
the qubits, so Q50's extra 48 don't hurt. → **Device choice is NOT driven by noise
here**; decide on availability / queue / quota and live calibration. Caveat: real
54q hardware may have more drift/crosstalk than the model shows, so on real metal
Q20 may still be safer — not claimable from models alone.

## Running takeaways

1. **Features are the ceiling** — scaling/encoding/training move within/near the noise band.
2. **Honest headline:** *tuned* classical (XGBoost 0.190) wins on HI-Small; quantum
   angle (0.125) is real signal (> RBF, > random) but behind the tree.
3. **Hardware:** angle encoding — the only cleanly NISQ-suitable option.
4. **Open lever:** distribution-shaping scalers (sweep #4).
