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

## 6 · REAL HARDWARE — VTT Q50, third vendor (2026-10-09) ⭐⭐⭐

**Conditions:** same frozen angle-kernel batch as #5 (`data/hw_batch.json`, 6
qubits, test × support-vector overlap circuits, 1024 shots), run on the **real
VTT Q50** via LUMI (SLURM job on the `small` partition, QPU submitted through the
LUMI-Q interface). Wall time 358 s (most of it QPU-queue polling).

| backend | vendor | qubits | PR-AUC sim → hw | drop | drift |
|---|---|---|---|---|---|
| ibm_marrakesh (REAL) | IBM | 156 | 0.111 → 0.111 | 0.000 | 0.007 |
| Garnet (REAL) | IQM | 20 | 0.111 → 0.111 | 0.000 | 0.010 |
| **Q50 (REAL)** | **VTT** | **50** | 0.111 → **0.100** | **+0.011** | **0.018** |

**vs last time:** third real QPU, third vendor. Q50 is the **first real chip
where hardware noise visibly touches the result** — drift 0.018 (2–3× IBM/Garnet)
and a 0.011 PR-AUC drop. **Finding:** even on the noisiest of the three devices,
the shallow angle encoding (depth 2, 0 two-qubit gates) only costs one PR-AUC
point, and 0.100 is still above the ~0.093 floor. Note Q50's *real* drift (0.018)
lands between the earlier **Q50 noise-model** guess (0.033, entry #5) and the
clean chips — the model over-stated it, consistent with #2's "models are
optimistic/pessimistic, trust the metal" caveat. Degradation report is now
**three vendors: two zero-drop, one −0.011.**

## 7 · Hybrid cascade — classic-first, quantum-enhance (2026-10-09)

**Conditions:** `hybrid.py`. XGBoost scores every test window; the **ambiguous**
slice — classical P(suspicious) within ±0.15 of 0.5 — is re-scored by the angle
QSVM and folded back inside the band. Confident classical calls are untouched.
Real data, `train_cap=200`, `test_cap=3000`, seed 0 (single seed).

**Result**

| stage | PR-AUC | FP-elim @90% rec |
|---|---|---|
| classical (XGBoost) alone | 0.198 | 21.7% |
| + quantum-enhanced ambiguous slice | 0.196 | 21.7% |

93 of 3,000 windows were ambiguous (24 of them suspicious) → routed to the QSVM.

**vs last time:** first hybrid run, and the last lever we hadn't pulled. **Finding:**
sending only the *uncertain* windows to the quantum kernel does **not** beat
classical alone (−0.002, inside the noise band). This confirms the benchmark from
the architecture side: the quantum kernel holds no signal the tree is missing on
this feature set. "Classical-first, quantum-enhanced" is a sound, swappable design
— it just doesn't move the number on HI-Small today. **Features remain the ceiling.**

## 8 · Bandwidth tuning — quantum pulls LEVEL with the tree (2026-10-09) ⭐

**Conditions:** `improve.py` / `improve_verify.py`. Angle encoding, real data,
`train_cap=200`, **identical splits for every model**, seed-averaged. Bandwidth
`c` = a global multiplier on the [0,π] encoding angles (c=1 = the old baseline);
smaller c fights the fidelity kernel's exponential concentration (Shaydulin &
Wild 2021). Projected QK = per-qubit ⟨X⟩,⟨Y⟩,⟨Z⟩ expectations → classical RBF
(Huang et al. 2021).

**Result (`improve_verify.py`, 5 seeds, test=3000, same splits):**

| method | PR-AUC |
|---|---|
| **angle c=0.15** | **0.184 ± 0.042** |
| angle c=0.10 | 0.181 ± 0.051 |
| angle c=0.05 | 0.173 ± 0.049 |
| **XGBoost (tuned)** | **0.167 ± 0.030** |
| angle c=0.20 | 0.159 ± 0.043 |
| angle c=0.30 | 0.152 ± 0.038 |
| projected QK | 0.137 ± 0.015 |
| **angle c=1.0 (OLD baseline)** | **0.115 ± 0.017** |
| RBF (tuned) | 0.097 ± 0.009 |

**vs last time (the big correction):** the benchmark (#3) ran the quantum kernel at
bandwidth **c=1 ([0,π]), where the fidelity kernel concentrates** → 0.115–0.125.
Shrinking the angles to **c≈0.1–0.15** lifts it to **~0.18 — statistically level
with the tuned XGBoost tree (0.167)**: overlapping error bars, mean marginally
above. The projected kernel gives a tighter (±0.015) lift to 0.137.
**Caveats:** (1) small-c variance is high (±0.04–0.05) → this is a **tie, not a
win**; (2) same-split XGBoost here is 0.167, below the deck's 3-seed 0.190 —
classical is itself noisy at train=200. Scaling sweep #4 under-explored this
(stopped at c=0.5). **Revised headline: once the quantum kernel's bandwidth is
tuned, quantum and classical are statistically indistinguishable on HI-Small —
the earlier "classical clearly wins" was partly an untuned-bandwidth artifact.**

## 9 · Encoding × bandwidth — angle stays best, even tuned (2026-10-09)

**Conditions:** `encoding_bandwidth.py`. Angle vs ZZ-linear, each swept across
bandwidth `c`, same splits, 3 seeds, test=1500. (Amplitude excluded — it
normalises the state, so bandwidth has no effect.)

**Result:**

| encoding @ best c | PR-AUC | at c=1.0 (old) |
|---|---|---|
| **angle c=0.05–0.1** | **0.23 ± 0.05** | 0.131 |
| ZZ-linear c=0.05 | 0.21 ± 0.04 | 0.117 |
| XGBoost (ref) | 0.21 ± 0.03 | — |
| RBF | 0.104 | — |

**Finding (answers "did we bandwidth-tune other encodings?"):** no, we hadn't —
and doing it now lifts **both** encodings (the encoding sweep #1 judged them all at
the crippling c=1). But **angle stays the best encoding even when ZZ is
bandwidth-tuned** (0.23 vs 0.21), and ZZ **concentrates faster** — peak at c=0.05,
collapsed by c=0.1 — as the entangling-gate theory predicts. So the angle pick is
vindicated twice: best simulator score **and** the only NISQ-safe encoding (0
two-qubit gates). **Caveat:** small-c variance is high and test-size-sensitive —
this 3-seed/1500 run reads ~0.23; the stricter 5-seed/3000 verify (#8) reads ~0.18.
Use the conservative **~0.18 = level with the tree**, never the rosy 0.23.

## 10 · No-trial-and-error bandwidth + noise-model check + FPR (2026-10-09)

Three questions settled: (a) can we pick bandwidth *without* sweeping? (b) does
the tuned kernel survive hardware noise? (c) what is our FPR, really?

**(a) Principled bandwidth selection** (`bandwidth_select.py`, 3 seeds, angle, 12q):

| selector | predicted c* | PR-AUC there | verdict |
|---|---|---|---|
| **Geometric rule `1/√(n·d)`** | **0.083** | **0.232** | ✅ lands on the true peak (c=0.08 → 0.232) |
| Kernel-Target Alignment (KTA) | 1.000 | 0.131 | ❌ picked the worst bandwidth |

**Finding:** the **geometric rule nails it analytically — zero compute, no sweep**
(answers Meeri's "mostly trial-and-error"). **KTA failed** on this data: with 9.3%
prevalence the `yyᵀ` target is dominated by negative-negative pairs, so centered
alignment doesn't track what the SVM uses (it rose monotonically toward c=1, the
opposite of PR-AUC). Would need an imbalance-weighted KTA; not worth it when the
geometric rule already hits the optimum. **Use `c = 1/√(n·d)` as the default.**

**(b) Noise-model check** (`export_hw_batch --bandwidth` → `hardware_batch --backend
apollo`, 6 qubits, 40 test, NO credits):

| config | sim PR-AUC | Apollo (noisy) | drift |
|---|---|---|---|
| c=1.0 (old) | 0.135 | 0.135 | 0.021 |
| c=0.1 (tuned) | 0.292 | 0.314 | **0.256** |

**Finding:** the bandwidth gain **survives the noise model** (0.31 vs 0.14). Drift is
huge (0.256 ≈ the 6-qubit readout floor) because at small c all overlaps sit near 1
and readout noise drags them down *uniformly* — and a uniform shift **cancels** in
the SVM decision (`Σ dual = 0`), so the ranking/PR-AUC is preserved. **Caveats:**
fake models understate real noise; margin is thin (at the floor); c=0.1 is
near-degenerate (all 80 train points became support vectors → 3,200 circuits,
~2× the c=1 cost).

**(c) FPR trade-off** (`fpr_tradeoff.py`, test=3000, prevalence 9.3%):

| operating point | XGBoost FPR | quantum c=0.1 FPR | precision | recall |
|---|---|---|---|---|
| @ 90% recall | 80.8% | 83.4% | ~10% | 90% |
| @ ~77% precision | 0.1% | 0.1% | ~77% | 2.5% |

**Finding:** FPR is meaningless without recall. FD4QC's headline "QSVM FPR 1.36% @
77% precision" is a **low-recall** point — and at that same precision *our* FPR is
~0.1% with recall ~2.5%. Any model hits ~1% FPR by barely flagging anything; the
real question is FPR *at a useful recall*, where everything is ~80%+ (feature
ceiling). The low-FPR "auto-clear" framing is the right **product** story, but it is
**not** a quantum-specific win here — quantum ≈ classical at matched points.

## 11 · REAL HARDWARE — bandwidth-tuned kernel on 3 QPUs: Q50 + Garnet + IBM (2026-10-09) ⭐⭐⭐

**Conditions:** the **bandwidth-tuned** frozen batch (`export_hw_batch --bandwidth
0.1 --n-train 40 --n-test 40`): 6 qubits, 40 support vectors × 40 test = **1,600
circuits**, 1,024 shots, on the **real VTT Q50** via LUMI (`run_q50.sh`). Wall time
**1,210 s** (~20 min). This is the FIRST real-hardware run of the *improved* config
(entries #5–6 were the untuned c=1 batch).

| config | device | PR-AUC sim → hw | drop | drift | circuits |
|---|---|---|---|---|---|
| c=1.0 (untuned) | Q50 | 0.111 → 0.100 | +0.011 | 0.018 | 656 |
| c=1.0 (untuned) | Garnet | 0.111 → 0.111 | 0.000 | 0.010 | 656 |
| **c=0.1 (tuned)** | **Garnet** | **0.231 → 0.183** | **+0.049** | **0.040** | 1,600 |
| **c=0.1 (tuned)** | **Q50** | **0.231 → 0.265** | **−0.033** | **0.116** | 1,600 |
| **c=0.1 (tuned)** | **ibm_fez** | **0.231 → 0.395** | **−0.164** | **0.033** | 1,600 |

*(All three tuned runs are full 1,024 shots. An earlier IBM 512-shot stopgap, 0.216, is dropped.)*

**Finding:** the bandwidth-tuned kernel **survives real Q50 hardware with no
degradation** — hardware PR-AUC (0.265) matches/slightly exceeds the simulator
baseline (0.231). The negative drop is within the noise of a 40-point test (read as
"held," not "improved"). Drift rose to **0.116** (~6× the c=1 runs) because the
small-angle kernel has less contrast and sits nearer the readout floor — **but the
score holds**, because readout noise is ~uniform and cancels in the SVM decision
(`Σ dual = 0`). Real-Q50 drift (0.116) came in **well below** the Apollo noise-model
guess (0.256), consistent with #2/#6: models are pessimistic, trust the metal.
**Directionally, bandwidth tuning roughly doubled the real-hardware score on Q50
(0.100 → 0.265)** — though the two batches differ (16- vs 40-test, c=1 baseline
0.111 vs tuned 0.231), so treat it as a strong directional result, not a strict
apples-to-apples.

**Three-vendor result:** the tuned kernel ran on **all three real QPUs**:
Garnet 0.183 (drop +0.049, drift 0.040, 606.9 s), ibm_fez 0.216 (drop +0.016,
drift 0.080, 251.8 s, 512 shots, single job), Q50 0.265 (drop −0.033, drift 0.116,
1,210 s). The three hardware scores **straddle** the sim baseline —
`0.183 < 0.216 < 0.231(sim) < 0.265` — averaging **0.221 ≈ 0.231**, i.e. **"held on
average" across three vendors**. IBM was the cleanest (closest to sim, −0.016).
**What we learned:** (1) the tuned kernel is hardware-viable on 3 vendors, not a sim
artifact; (2) but it is **noisier than the untuned c=1 config** (which landed
0.100/0.111 — near-identical on both chips) — the price of the smaller-contrast
kernel sitting nearer the readout floor (drift 0.04–0.12 vs 0.007–0.018 at c=1);
(3) **drift does not predict PR-AUC at this test size** — Garnet had the lowest
drift (0.040) yet the biggest drop, because with 40 test points PR-AUC noise (±0.05)
dominates; (4) the score survives because readout noise is ~uniform and cancels in
the SVM decision (`Σ dual = 0`), but that cancellation is device-dependent (clean on
Q50, partial on Garnet). **Defensible claim:** the robust hardware story is still
the **c=1 config** (near-zero drift, 3 vendors). The **tuned config is "runs and
holds on average on 3 QPUs"** — promising, honestly noisier, and under-powered (would
need more test points/seeds to tighten, which costs hardware time). **Caveats:**
single seed; 40 test points → noisy PR-AUC; IBM at 512 shots (others 1,024).

**UPDATE — IBM full-shot run exposes the tuned config's instability (2026-10-10):**
re-ran the identical tuned batch on **ibm_fez at 1,024 shots** (482.7 s): sim 0.231
→ **0.395**, drift **0.033**. A hardware PR-AUC *above the noiseless simulator* cannot
be a real signal — it is noise that happened to reorder the 40 test points
favorably. **The tell:** drift is *low* (0.033, kernel barely perturbed) yet PR-AUC
swung huge, because the tuned batch is **near-degenerate — all 40 training points
are support vectors**, so the SVM's large, opposing dual coefficients *amplify* tiny
readout noise into big decision swings (`noise · dual` has high variance even when
each kernel entry is close). Same batch, three full-shot runs (one per vendor)
scatter **0.183 / 0.265 / 0.395** around sim 0.231 — a **0.18–0.40 spread**. **Revised conclusion:** the
tuned config's hardware PR-AUC is **noise-dominated at this scale — do NOT report a
single tuned-hardware number (or an "average") as a result.** The honest, stable
hardware validation is the **robust c=1 config** (0.111→0.111 on 3 vendors, drift
0.007–0.018). The tuned run is a *finding about fragility*: aggressive bandwidth
tuning buys simulator score but makes the on-device readout unstable. **This
supersedes the "held on average 0.221" line above — that used the 512-shot IBM run;
the full-shot run breaks it.**

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
2. **Honest headline (updated #8):** at the *right bandwidth* (c≈0.1–0.15) the
   quantum angle kernel reaches ~0.18, **statistically level with tuned XGBoost
   (~0.17)** — a tie within noise, not a win. The old "classical clearly wins
   (0.190 vs 0.125)" compared a *tuned* tree against an *untuned-bandwidth* kernel.
3. **Encoding:** angle wins **twice** — best simulator score even vs a
   bandwidth-tuned ZZ (#9), *and* the only cleanly NISQ-suitable option;
   **validated on 3 real QPUs / 3 vendors** (IBM, IQM, VTT) — two zero-drop,
   Q50 −0.011 (drift 0.018). Shallow encoding holds even on the noisiest chip.
4. **Open lever:** distribution-shaping scalers (sweep #4).
