# Q-SIEVE — code walkthrough for a new quantum learner

A guided tour of everything we built. Read top to bottom; the quantum parts are
explained line by line. No prior quantum knowledge assumed.

---

## 0. The one-paragraph summary

Q-SIEVE helps a bank's anti-money-laundering (AML) analyst. The bank already
*flags* account-weeks; about 388,000 of them, and about 99% are innocent. The
table we score keeps all 4,084 suspicious windows and 40,000 innocent ones, so
random PR-AUC is 0.093. Inside it, a **quantum kernel** does one job: measure
how *similar* two windows are. Untuned, the tree wins. After bandwidth, quantum
is level with the tree. The robust 6-qubit kernel ran on three real machines.
Safe auto-clear, at a 1% error bar, clears nothing. The honesty *is* the pitch.

**The headline numbers** (PR-AUC — explained in §7; random guessing = 0.093):
- Untuned fight: **XGBoost 0.190**, quantum angle **0.125**, RBF **0.097**
- After bandwidth, 5 seeds, 3,000 test windows: quantum **0.18 ± 0.04**, tree **0.17 ± 0.03** — a tie
- Robust hardware, 6 qubits: Garnet **0.111 → 0.111**, IBM Marrakesh **0.111 → 0.111**, Q50 **0.111 → 0.100**
- Tuned hardware scatters **0.183 / 0.265 / 0.395** around a 0.231 simulator. The 0.395 is noise.

---

## 1. The three layers (the mental model)

```
A flagged account-window  (e.g. account X, one week of transactions)
        │
        ▼
 Layer 1  — Consolidation   turn messy transactions into ~12 numbers ("features")
        │                    consolidate.py
        ▼
 Layer 2  — Compare         how similar are two windows?   ← the QUANTUM part
        │                    kernels.py / quantum_kernel.py (or baseline.py = classical)
        ▼
 Layer 3  — Trust layer     calibrate, auto-clear the confident-clean, abstain,
        │                    audit, explain   trust.py
        ▼
 Decision:  "auto-clear"  or  "leave in the queue for a human"
```

Only Layer 2 is quantum. Everything else is ordinary Python.

---

## 2. Repo structure (what each file is for)

**Data**
- `data.py` — loads the feature table (real or synthetic) + `cap_train` (subsampling)
- `demo_data.py` — makes fake data so the app runs with no dataset
- `build_features.py` — turns the raw IBM AML CSV into `data/X.csv`

**Layer 1 — features**
- `consolidate.py` — one window → 12 behavioural numbers

**Layer 2 — the kernel (quantum) + classical**
- `kernels.py` — ⭐ every quantum kernel variant, one place (fixed / trained / amplitude)
- `quantum_kernel.py` — a scikit-learn-style wrapper so the quantum kernel "looks like" a normal model
- `baseline.py` — the classical comparison (XGBoost + RBF)

**Layer 3 — the product**
- `trust.py` — calibration, auto-clear, audit log, precedent, attribution

**Glue & evaluation**
- `pipeline.py` — ties the layers together (pick classical or quantum)
- `metrics.py` — the scoring functions (PR-AUC helpers, FP-at-recall, noise-aware ranking)
- `sweep.py` — compares all encodings
- `benchmark.py` — fair comparison vs tuned classical
- `scaling_sweep.py` — tests feature scaling
- `experiments.py` — the two headline charts

**App**
- `app.py` — the Streamlit analyst dashboard

**Quantum demos & hardware**
- `quantum_demo.py` — ⭐ "hello world": one kernel, step by step
- `q20_noise_sweep.py`, `q_device_compare.py` — noise analysis on device models
- `garnet_connect.py`, `ibm_connect.py` — no-cost connection checks
- `export_hw_batch.py` — freeze a small model for hardware
- `hardware_batch.py` (IQM), `ibm_batch.py` (IBM) — run on real quantum computers

**Results**
- `RESULTS.md` — the lab notebook (every experiment, dated)
- `results/*.csv` — raw result snapshots

**Three virtual environments** (isolated so their libraries don't clash):
- `.venv` — the ML stack (qiskit, scikit-learn, xgboost, streamlit)
- `.venv-iqm` — IQM hardware (`iqm-client[qiskit]`)
- `.venv-ibm` — IBM hardware (`qiskit-ibm-runtime`)

---

## 3. THE QUANTUM CORE — `quantum_demo.py` line by line

This is the file to understand first. It shows what Layer 2 actually does.

```python
fmap = zz_feature_map(feature_dimension=2, reps=1, entanglement="linear")
```
**Build the "feature map."** A feature map is a recipe that turns your numbers
into a quantum circuit. `feature_dimension=2` = two numbers → two qubits. The
numbers become **rotation angles** for quantum gates. (That's why we scale
features into `[0, π]` — they're angles.)

```python
state_A = Statevector(fmap.assign_parameters(A))
```
**Encode one window into a quantum state.** `assign_parameters(A)` plugs window
A's numbers into the circuit; `Statevector(...)` computes the resulting quantum
state. For 2 qubits that's 4 complex numbers (amplitudes). *The data now lives in
a quantum state* — this is "quantum encoding."

```python
kernel = FidelityStatevectorKernel(feature_map=fmap)
K = kernel.evaluate([A, B, C])
```
**The one quantum job: similarity.** "Fidelity" = the overlap between two quantum
states, `|⟨φ(x)|φ(z)⟩|²`. It's 1.0 if identical, 0.0 if completely different.
`K` is a grid of these similarities between windows A, B, C. **This matrix is the
ONLY thing the quantum computer produces.** A normal classical SVM (support
vector machine) then draws the decision boundary from it.

```python
overlap.compose(fmap.assign_parameters(A), inplace=True)
overlap.compose(fmap.assign_parameters(B).inverse(), inplace=True)
overlap.measure_all()
```
**How real hardware estimates that number.** You can't read amplitudes off a real
chip. So you build the circuit `U(B)† · U(A)` (encode A, then *un-encode* B) and
measure. If A and B are identical the un-encoding perfectly cancels and you
always measure all-zeros; the more different they are, the less often you get
all-zeros. So **the fraction of all-zero measurements ≈ the fidelity**. This is
exactly how `hardware_batch.py` computes the kernel on Garnet/IBM.

**Run it:** `.\.venv\Scripts\python.exe quantum_demo.py`

---

## 4. `kernels.py` — all the kernel variants (the heart of Layer 2)

One file, three clearly-labelled sections. The key functions:

```python
def feature_map(encoding, d, reps=1, entanglement="linear"):
    if encoding == "angle":
        return z_feature_map(feature_dimension=d, reps=reps)   # single-qubit rotations, NO entanglement
    if encoding == "zz":
        return zz_feature_map(feature_dimension=d, reps=reps, entanglement=entanglement)  # adds entangling gates
```
- **angle** (`z_feature_map`): each feature = one rotation on one qubit. Shallow,
  no two-qubit gates → **survives noise best**. This is our hardware choice.
- **zz** (`zz_feature_map`): adds *entangling* gates between qubits (richer, but
  deeper → noisier). The Qiskit default; underperformed on our data.

```python
def fixed_kernel(encoding, d, reps=1, entanglement="linear"):
    return FidelityStatevectorKernel(
        feature_map=feature_map(encoding, d, reps, entanglement),
        cache_size=CACHE, auto_clear_cache=False)       # cache = don't recompute states
```
Builds the similarity kernel from a feature map. `cache_size` stores computed
quantum states so repeated scoring is fast (this is what took our dashboard from
~30s to 0.8s).

```python
def amplitude_kernel(A, B):
    def prep(M):
        n = 1 << int(np.ceil(np.log2(M.shape[1])))      # pad to a power of 2
        P = np.zeros((len(M), n)); P[:, :M.shape[1]] = M
        return P / np.clip(np.linalg.norm(P, axis=1, keepdims=True), 1e-12, None)  # normalize
    return (prep(A) @ prep(B).T) ** 2                    # squared overlap
```
**Amplitude encoding**, computed exactly without a circuit. It's just the
*squared cosine similarity* of normalized vectors. Great on a simulator, but on
real hardware it needs a *deep* state-prep circuit → terrible on NISQ (we
measured 15× worse noise). So: simulator-only.

```python
def train_zz_scales(Xtr_s, ytr, entanglement="linear", maxiter=40):
    fm, theta = trainable_zz_circuit(Xtr_s.shape[1], entanglement)   # ZZ with tunable weights
    qk = TrainableFidelityStatevectorKernel(feature_map=fm, training_parameters=list(theta), ...)
    res = QuantumKernelTrainer(quantum_kernel=qk, loss=SVCLoss(class_weight="balanced"),
                               optimizer=SPSA(maxiter=maxiter, blocking=True),
                               initial_point=np.full(len(theta), 2.0)).fit(Xtr_s, ytr)
    return res.quantum_kernel
```
**Kernel training** (Meeri's idea). Instead of a *fixed* feature map, add free
weights `theta` and let an optimizer (`SPSA`) tune them to separate the classes
better. `initial_point = 2.0` starts from the fixed map (so training can only
help). Honest result: it *didn't* help on our data.

---

## 5. `quantum_kernel.py` — making quantum "look like" a normal model

```python
class QuantumKernelClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.scaler_ = MinMaxScaler((0, np.pi)).fit(X)        # features → angles
        self.X_train_ = self.scaler_.transform(X)
        self.kernel_ = build_kernel(X.shape[1], ...)          # the quantum kernel
        Ktr = self.kernel_.evaluate(self.X_train_)            # train similarity grid
        self.svc_ = SVC(kernel="precomputed", ...).fit(Ktr, y)  # classical SVM on it
    def decision_function(self, X):
        return self.svc_.decision_function(self.similarity(X))
```
This wraps the quantum kernel so it has the same `.fit()/.predict()` interface as
any scikit-learn model. That's *why* Layer 3 can swap classical ↔ quantum with
one word — the trust layer never knows it's talking to a quantum computer.

---

## 6. `baseline.py` and `trust.py` — the classical comparison and the product

**baseline.py** — `train_baselines()` fits **XGBoost** (a gradient-boosted tree)
and an **RBF-SVC** (classical kernel), both weighted for class imbalance
(`scale_pos_weight` / `class_weight="balanced"`) so they don't just predict
"legitimate" for everything. These are what quantum is measured against.

**trust.py** — the actual deliverable:
- `calibrate()` — makes a score of 0.9 really mean "90% likely," using isotonic regression.
- `auto_clear_curve()` — sweeps a threshold; for each, how much of the queue we
  clear vs how often we're wrong. **The business chart.**
- `AuditLog` — records every decision (reproducible later).
- `precedents()` — "this resembles 3 confirmed cases" (reuses the kernel similarity).
- `attribution()` — which feature drove this decision.

The safety rule, in code: the system only ever *clears*; it never closes a
suspicious case. So the one risk is a wrong auto-clear, which is why
`choose_threshold()` picks conservatively (error among cleared ≤ 1%).

---

## 7. The numbers — what we compared and why each one

**The metric: PR-AUC** (precision-recall area under curve). We do NOT use accuracy
because at 9% suspicious, "predict all legitimate" is 91% accurate and catches
zero launderers. PR-AUC focuses on the rare positive. Reference points:
- **0.093** = random guessing (= the fraction that are suspicious)
- **1.0** = perfect

**The fair benchmark** (`benchmark.py`, 12 features, tuned, identical splits):

| Model | PR-AUC | What it is |
|---|---|---|
| XGBoost (tuned) | **0.190** | classical tree — wins the untuned fight |
| Quantum (angle, c=1) | 0.125 | untuned kernel — beats RBF, loses to the tree |
| Quantum (angle, c≈0.1) | **0.18 ± 0.04** | bandwidth-tuned — level with the tree (0.17 ± 0.03), a tie |
| RBF (tuned) | 0.097 | classical kernel |
| random | 0.093 | floor |

*Why these numbers:* the untuned kernel sits close to the 0.093 floor because
features were rotations across the full range, so every pair looked equally
different. Shrinking that bandwidth is the knob that moved the score, from
0.125 to about 0.18, level with the tree. Encoding, circuit training, and a
hybrid did not. The features are still a low ceiling. The tie is not a win.

**What we compared across experiments:**
- `sweep.py` — encodings (ZZ / angle / amplitude / trained / PCA). → amplitude &
  angle beat ZZ; training didn't help.
- `scaling_sweep.py` — scalers (MinMax / log / quantile / …). → shapers nudge
  quantum up ~0.02 but within noise.
- `q20_noise_sweep.py` — how each encoding survives device noise. → angle is the
  only cleanly NISQ-safe choice (drift 0.010 vs amplitude 0.158).

---

## 8. The hardware runs — `export_hw_batch.py` → `hardware_batch.py` / `ibm_batch.py`

**`export_hw_batch.py`** freezes a tiny model for hardware:
```python
sel = SelectKBest(f_classif, k=D_HW).fit(Xtr, ytr)   # keep the 6 strongest features
...
svc = SVC(kernel="precomputed", ...).fit(kernel.evaluate(Atr), ytr)
sv = svc.support_                                     # the support vectors
```
We use **6 features = 6 qubits** (not 12) because the hardware kernel measures
*all qubits as all-zeros*, and readout error compounds per qubit — 12 qubits
would double the noise floor. It saves only the **support vectors** (the few
training points the SVM actually uses) + weights, so the hardware only has to
compute `test × support` similarities (~656 small circuits).

**`hardware_batch.py`** runs those overlap circuits on a backend:
```python
counts = backend.run(chunk, shots=...).result().get_counts()   # measure
K = [c.get("0"*d, 0) / shots ...]                               # all-zeros fraction = kernel entry
dec = K @ dual + intercept                                      # rebuild the SVM decision
```
It chunks into ≤100 circuits/job (both IQM and IBM cap a job at 100), then
reconstructs the decision and reports **simulator PR-AUC vs hardware PR-AUC** plus
the **drift** (how far hardware entries move from exact).

**The result** (two vendors, same frozen batch):

| Device | PR-AUC sim → hw | drift |
|---|---|---|
| IQM Garnet (20q) | 0.111 → 0.111 | 0.010 |
| IBM marrakesh (156q) | 0.111 → 0.111 | 0.007 |

**Zero degradation on both.** Angle encoding is so shallow the noise barely
touches it. (The drift difference is within run-to-run noise — *not* because IBM
has more qubits; we only use 6 on each.)

---

## 9. What we did tonight (the story in order)

1. Scaffolded the repo + froze the "feature contract."
2. Built Layer 1 (features), the classical baseline, and Layer 3 (trust layer).
3. Built a Streamlit dashboard.
4. Got the real IBM AML data (5M transactions), currency-normalized, added a
   velocity-vs-baseline feature → 12 features.
5. Built the quantum kernel (Qiskit) and compared encodings, scaling, training —
   all disciplined with seed-repeats and noise-aware ranking.
6. Did a **fair** benchmark vs *tuned* XGBoost/RBF → classical wins honestly.
7. Analysed NISQ noise → chose **angle encoding** for hardware.
8. Ran the robust 6-qubit kernel on **three real quantum computers** (IQM Garnet, IBM Marrakesh, VTT Q50): 0.111, 0.111, 0.100.
9. Cleaned the code into one-responsibility-per-file and logged every result.

---

## 10. How to run everything

```powershell
# the ML stack
.\.venv\Scripts\python.exe quantum_demo.py            # see one kernel run
.\.venv\Scripts\python.exe sweep.py                   # compare encodings
.\.venv\Scripts\python.exe benchmark.py               # fair vs tuned classical
.\.venv\Scripts\python.exe -m streamlit run app.py    # the dashboard

# hardware (isolated venvs)
.\.venv-iqm\Scripts\python.exe q20_noise_sweep.py                 # noise analysis (local)
.\.venv-iqm\Scripts\python.exe hardware_batch.py --backend garnet # real IQM (spends credits)
.\.venv-ibm\Scripts\python.exe ibm_batch.py                       # real IBM (free allocation)
```

---

## 11. The honest one-liner for the jury

> *"Untuned, the tree wins, 0.19 to 0.12. Bandwidth brings the quantum kernel
> level with the tree, 0.18 to 0.17, a tie. The robust kernel held on three real
> machines: 0.111, 0.111, and 0.100. At a safe bar, both models clear zero."*

That honesty, backed by real-hardware results, is the strongest thing you can
show people who know this field.
