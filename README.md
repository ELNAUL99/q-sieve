# Q-SIEVE

A trust layer for anti-money-laundering alert triage. The quantum piece computes
how similar two flagged windows are. A classical model ranks. The trust layer
may only auto-clear a window it is confidently sure is legitimate. It never
accuses, and it never closes a suspicious case.

On IBM's public AML set there are about 388,000 flagged account-weeks, and
about 99% of them are innocent. The table we score keeps all 4,084 suspicious
windows and 40,000 innocent ones, so a random rank scores 0.093. Untuned, the
tree leads (0.19 vs 0.12). After bandwidth, quantum is level with the tree
(0.18 ± 0.04 vs 0.17 ± 0.03), a tie. At a safe error bar, both clear zero
windows. The robust 6-qubit kernel ran on IQM Garnet, IBM Marrakesh, and VTT
Q50: 0.111, 0.111, and 0.100.

Live demo (the pitch dashboard): https://elnaul99.github.io/q-sieve/

Submission for the Quantum × Finance Hackathon (Hanken).

## The three layers

| Layer | Owner | Tech | What it does |
|-------|-------|------|--------------|
| 1 — Consolidation | pipeline | pandas | window -> ~12 behavioural features |
| 2 — Quantum kernel | quantum | Qiskit | how similar are two windows? |
| 3 — Trust layer | pipeline | scikit-learn | calibrate, abstain, auto-clear, explain, log |

## The feature-table contract (`consolidate.py`)

12 features per flagged `(account, window)`:

| # | column | meaning |
|---|--------|---------|
| 1 | `n_tx` | transaction count |
| 2 | `frag_ratio` | gross ÷ count (low = smurfing) |
| 3 | `in_out_count` | credits ÷ debits (count) |
| 4 | `in_out_value` | credit ÷ debit value |
| 5 | `pass_through` | \|net\| ÷ gross (~0 = conduit/mule) |
| 6 | `fan_in` | distinct senders |
| 7 | `fan_out` | distinct recipients |
| 8 | `round_share` | fraction round-number amounts |
| 9 | `near_10k` | fraction near the 10k threshold |
| 10 | `near_1k` | fraction near 1k |
| 11 | `iat_entropy` | entropy of inter-arrival gaps |
| 12 | `velocity_ratio` | window tx-count ÷ the account's own baseline |

All amounts are normalized to USD before feature computation (the data mixes 15
currencies). `velocity_ratio` is cross-window, so it is added during table
assembly, not inside `consolidate()`.

- **Scaler (quantum):** `MinMaxScaler((0, π))` — features become rotation angles.
- **Split:** `train_test_split(stratify=y, test_size=0.3, random_state=0)`.
- **Metric:** PR-AUC (not accuracy, which is meaningless at this imbalance);
  business metric = false positives eliminated at 90% recall.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Tokens for IBM/IQM go in a local `.env` (copy `.env.example`). Never commit `.env`.

Rebuild the real data: download the IBM AML HI-Small dataset to
`data/HI-Small_Trans.csv`, then `python build_features.py` -> `data/X.csv`.

## Run

```bash
.venv/Scripts/python.exe quantum_demo.py            # see one kernel run
.venv/Scripts/python.exe sweep.py                   # compare encodings
.venv/Scripts/python.exe benchmark.py               # fair vs tuned classical
.venv/Scripts/python.exe -m streamlit run app.py    # local lab app
# pitch dashboard: q-sieve-demo/index-real.html, also the live link above
```

Hardware runs use the isolated `.venv-iqm` / `.venv-ibm` — see `QUANTUM_GUIDE.md`.

## References
- FD4QC — classical vs quantum fraud detection on IBM AML: https://arxiv.org/abs/2507.19402
- Havlíček et al. — quantum feature-map kernels: https://arxiv.org/abs/1804.11326
