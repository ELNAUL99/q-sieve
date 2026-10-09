"""Build model-data.real.json: the teammate's UI schema, filled from OUR real pipeline.

Real IBM AML windows, angle-kernel SVC scores, real precedents + transactions,
and the real benchmark + three-QPU hardware blocks. Keeps every field the UI reads.

Run from the repo root:  .venv/Scripts/python.exe q-sieve-demo/make_real_demo.py
(reads ../data/X.csv and ../data/HI-Small_Trans.csv)
"""
from __future__ import annotations
import os, sys, json, datetime as dt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import kernels
from data import load_features, cap_train
from baseline import split
from build_features import load_account_centric, RAW, MIN_TX

OUT = os.path.join("q-sieve-demo", "model-data.real.json")
TRAIN_CAP = 200
POOL_CAP = 3000
SEED = 0
WINDOW_HOURS = 168           # our windows are ISO calendar weeks
CLEAR_MAX, ESCALATE_MIN = 0.20, 0.75
N_CASES = 12

rng = np.random.default_rng(SEED)


def money(x): return float(round(x, 2))


def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))


# ---- 1. real modelled table + same split as the benchmark ----------------
X, y, meta, source = load_features()
assert source == "real", "need data/X.csv (real)"
feat_cols = list(X.columns)
Xtr0, Xte0, ytr0, yte0 = split(X, y)
Xtr, ytr = cap_train(Xtr0, ytr0, cap=TRAIN_CAP, seed=SEED)
tr_wids = meta.loc[Xtr.index, "window_id"].tolist()

# ---- 2. angle-kernel SVC (same config as benchmark) -----------------------
sc = MinMaxScaler((0, np.pi)).fit(Xtr)
Xtr_s = sc.transform(Xtr)
kern = kernels.fixed_kernel("angle", Xtr.shape[1], reps=1)
Ktr = kern.evaluate(Xtr_s)
svc = SVC(kernel="precomputed", class_weight="balanced", C=16).fit(Ktr, ytr)

# ---- 3. score a test pool, map decision -> riskScore ----------------------
Xpool, ypool = cap_train(Xte0, yte0, cap=POOL_CAP, seed=SEED)
pool_wids = meta.loc[Xpool.index, "window_id"].tolist()
Xpool_s = sc.transform(Xpool)
Kpool = kern.evaluate(Xpool_s, Xtr_s)           # (pool x train) fidelities
dec = svc.decision_function(Kpool)
z = (dec - dec.mean()) / (dec.std() + 1e-9)
risk = sigmoid(1.3 * z)                          # 0..1 decision score

# demo-cutoff (0.20) safety numbers for the quantum model, on this pool
below = risk < CLEAR_MAX
cov = float(below.mean())
err = float(ypool.to_numpy()[below].mean()) if below.any() else 0.0
n_below = int(below.sum()); n_susp_below = int(ypool.to_numpy()[below].sum())

# ---- 4. pick 12 windows spread across the score range ---------------------
order = np.argsort(risk)
targets = np.linspace(0.05, 0.97, N_CASES)
picks, used = [], set()
for t in targets:
    j = int(np.argmin(np.abs(risk - t)))
    k = 0
    while j in used:                              # avoid dupes
        k += 1
        j = int(np.argmin(np.abs(risk - t)) ) + k
        j = min(j, len(risk) - 1)
    used.add(j); picks.append(j)

FEATURES = [
    {"key": "velocity", "label": "Activity vs baseline", "unit": "×",
     "description": "Transactions this window divided by the account’s own median window size. High = a burst of activity."},
    {"key": "average_amount", "label": "Average transfer", "unit": "USD",
     "description": "Gross volume divided by transaction count, in USD (15 currencies normalised)."},
    {"key": "inbound_count_share", "label": "Inbound share (count)", "unit": "",
     "description": "Share of transactions that are credits into the account."},
    {"key": "inbound_value_share", "label": "Inbound share (value)", "unit": "",
     "description": "Share of gross value that flows in rather than out."},
    {"key": "net_to_gross", "label": "Net ÷ gross flow", "unit": "",
     "description": "Signed net change over gross flow. Near zero means money largely passes through."},
    {"key": "dwell_hours", "label": "Mean time between tx", "unit": "h",
     "description": "Average gap between consecutive transactions in the window."},
    {"key": "counterparties", "label": "Distinct counterparties", "unit": "",
     "description": "Number of distinct accounts transacted with this window."},
    {"key": "threshold_share", "label": "Near the 10k line", "unit": "",
     "description": "Share of transfers just under the 10,000 USD reporting threshold (9k–10k)."},
]

# ---- 5. load real transactions for the picked windows ---------------------
print("loading raw transactions (this takes a minute)…", flush=True)
long = load_account_centric(RAW)
pick_wids = [pool_wids[j] for j in picks]
sub = long[long["window_id"].isin(pick_wids)]
by_wid = {w: g for w, g in sub.groupby("window_id")}


def regime_of(ivs, ntg, thr):
    if ivs > 0.70: return "Inbound concentration"
    if ivs < 0.30: return "Outbound concentration"
    if abs(ntg) < 0.10: return "Pass-through / fragmented"
    if thr > 0.20: return "Near-threshold activity"
    return "Mixed activity"


def features_from_tx(g, vel):
    amt = g["amount"].to_numpy()
    inflow = float(amt[amt > 0].sum()); outflow = float(-amt[amt < 0].sum())
    gross = inflow + outflow; n = len(g)
    ts = np.sort(g["timestamp"].to_numpy())
    gaps = np.diff(ts).astype("timedelta64[s]").astype(float) / 3600.0 if n > 1 else np.array([0.0])
    dwell = float(np.clip(np.mean(gaps), 0, WINDOW_HOURS))
    aa = np.abs(amt)
    return {
        "velocity": float(vel),
        "average_amount": money(gross / max(n, 1)),
        "inbound_count_share": float((amt > 0).mean()),
        "inbound_value_share": float(inflow / gross) if gross else 0.0,
        "net_to_gross": float((inflow - outflow) / gross) if gross else 0.0,
        "dwell_hours": round(dwell, 2),
        "counterparties": int(g["counterparty"].nunique()),
        "threshold_share": float(np.mean((aa >= 9000) & (aa < 10000))),
    }, inflow, outflow, gross, n


def tx_rows(g, cap=120):
    g = g.sort_values("timestamp").head(cap)
    rows = []
    for i, r in enumerate(g.itertuples()):
        amt = float(r.amount)
        rows.append({
            "id": f"{r.window_id}-{i:03d}",
            "timestamp": pd.Timestamp(r.timestamp).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "direction": "in" if amt > 0 else "out",
            "amount": money(abs(amt)),
            "counterparty": str(r.counterparty),
        })
    return rows


# training-window summaries for precedents
tr_feat = X.loc[Xtr.index].reset_index(drop=True)
tr_lab = ytr.reset_index(drop=True)


def precedent_cards(pool_row_idx, k=3):
    sims = Kpool[pool_row_idx]                      # fidelity to each train window
    top = np.argsort(sims)[::-1][:k]
    cards = []
    for t in top:
        lab = int(tr_lab.iloc[t]); f = tr_feat.iloc[t]
        cards.append({
            "id": tr_wids[t].replace("|", "-"),
            "label": "labelled suspicious" if lab else "labelled legitimate",
            "similarity": float(sims[t]),
            "regime": regime_of(f["in_out_value"] if "in_out_value" in f else 0.5,
                                 0.0, f.get("near_10k", 0.0)),
            "summary": (f"{'Suspicious' if lab else 'Legitimate'} reference window · "
                        f"{int(f['n_tx'])} tx, velocity {float(f['velocity_ratio']):.1f}×."),
        })
    return cards


EV_DETAIL = {k: f["description"] for k, f in [(x["key"], x) for x in FEATURES]}


def evidence_cards(feats):
    def pct(v): return f"{v*100:.1f}%"
    cand = [
        ("net_to_gross", "Net ÷ gross flow", pct(feats["net_to_gross"])),
        ("velocity", "Activity above the account baseline", f"{feats['velocity']:.1f}×"),
        ("threshold_share", "Transfers near the 10k line", pct(feats["threshold_share"])),
        ("counterparties", "Counterparties this window", f"{feats['counterparties']}"),
        ("inbound_value_share", "Share of value inbound", pct(feats["inbound_value_share"])),
    ]
    return [{"feature": k, "title": title, "value": val, "detail": EV_DETAIL[k]}
            for k, title, val in cand[:4]]


cases = []
for rank, j in enumerate(picks):
    wid = pool_wids[j]
    g = by_wid.get(wid)
    if g is None or len(g) == 0:
        continue
    vel = float(Xpool.iloc[j]["velocity_ratio"])
    feats, inflow, outflow, gross, n = features_from_tx(g, vel)
    r = float(risk[j])
    status = "clear" if r < CLEAR_MAX else ("escalate" if r >= ESCALATE_MIN else "review")
    acct, week = wid.split("|")
    cases.append({
        "id": wid.replace("|", "-"),
        "regime": regime_of(feats["inbound_value_share"], feats["net_to_gross"], feats["threshold_share"]),
        "regimeIndex": rank,
        "profile": (f"Account …{acct[-6:]} · week {week} · {n} transfers · "
                    f"{feats['counterparties']} counterparties"),
        "count": int(n), "inflow": money(inflow), "outflow": money(outflow), "gross": money(gross),
        "features": feats,
        "label": int(ypool.iloc[j]),
        "split": "test",
        "name": f"Account …{acct[-6:]} · {week}",
        "riskScore": r,
        "status": status,
        "evidence": evidence_cards(feats),
        "precedents": precedent_cards(j),
        "truthNote": "Dataset laundering label for this window; shown for demo transparency, not an analyst input.",
        "transactions": tx_rows(g),
    })

cases.sort(key=lambda c: c["riskScore"], reverse=True)

# ---- 6. evaluation block: real benchmark + three real QPUs ----------------
models = [
    {"id": "xgboost", "name": "XGBoost (tuned)", "prAuc": 0.190,
     "recallAt90Policy": 0.90, "falsePositiveEliminationAt90Policy": 0.178,
     "autoClearCoverage": 0.0, "autoClearError": 0.0, "trainSeconds": 0.35, "testSeconds": 0.02},
    {"id": "quantum", "name": "Quantum kernel (angle) + SVC", "prAuc": 0.125,
     "recallAt90Policy": 0.90, "falsePositiveEliminationAt90Policy": 0.146,
     "autoClearCoverage": 0.0, "autoClearError": 0.0, "trainSeconds": 1.9, "testSeconds": 4.1},
    {"id": "rbf", "name": "RBF kernel (tuned)", "prAuc": 0.097,
     "recallAt90Policy": 0.90, "falsePositiveEliminationAt90Policy": 0.115,
     "autoClearCoverage": 0.002, "autoClearError": 0.01, "trainSeconds": 0.2, "testSeconds": 0.05},
]
hardware = {
    "status": "Run on real quantum hardware",
    "note": ("Same frozen angle-kernel batch (6 features → 6 qubits, to keep readout "
             "noise low) executed on three vendors’ real QPUs. PR-AUC here is the "
             "6-qubit hardware batch (exact sim baseline 0.111), not the 12-feature "
             "simulator benchmark above."),
    "devices": [
        {"name": "IBM marrakesh", "vendor": "IBM", "qubits": 156, "praucSim": 0.111, "praucHw": 0.111, "drift": 0.007},
        {"name": "IQM Garnet", "vendor": "IQM", "qubits": 20, "praucSim": 0.111, "praucHw": 0.111, "drift": 0.010},
        {"name": "VTT Q50", "vendor": "VTT", "qubits": 50, "praucSim": 0.111, "praucHw": 0.100, "drift": 0.018},
    ],
}
evaluation = {
    "counts": {"train": int(len(ytr)), "calibration": 3, "test": POOL_CAP,
               "demoPool": POOL_CAP, "positiveTrain": int(ytr.sum()),
               "positiveCalibration": 0, "positiveTest": int(ypool.sum())},
    "models": models,
    "hardware": hardware,
    "xgboost": {"status": "real", "note": "Real tuned XGBoost (GridSearchCV), not a substitute."},
    "demoCutoff": {"threshold": CLEAR_MAX, "coverage": round(cov, 4), "error": round(err, 4),
                   "belowCount": n_below, "suspiciousBelow": n_susp_below},
    "protocol": ("Real IBM AML HI-Small. Identical 12 features, 70/30 split (seed 0), "
                 "train capped at 200 for every model. Benchmark PR-AUC is mean over 3 seeds."),
    "caveats": [
        "Real data, honest result: tuned classical XGBoost leads on accuracy; the quantum kernel is real signal (beats the RBF kernel) but behind the tree.",
        "Features are the ceiling — scaling, encoding and kernel training all move within the noise band.",
        "Auto-clear at ≤1% error clears 0 windows on this data; the 0.20 cutoff shown in the queue is an interactive illustration, not a safe production policy.",
    ],
}

meta = {
    "title": "Q-SIEVE",
    "modelVersion": "qsieve-real-angle-v1.0",
    "generatedAt": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "seed": SEED,
    "dataMode": "Real IBM AML HI-Small transactions (public dataset)",
    "qubits": int(Xtr.shape[1]),
    "reps": 1,
    "entanglement": "none (angle encoding, 0 two-qubit gates)",
    "simulation": "Exact statevector simulator for the benchmark; real QPUs for the hardware batch.",
    "kernel": "K(x,z) = |⟨φ(x)|φ(z)⟩|²",
    "thresholdAmount": 10000,
    "currency": "USD",
    "windowHours": WINDOW_HOURS,
    "scoreDescription": "Sigmoid-scaled angle-kernel SVC decision score. Not a calibrated probability of financial crime.",
    "fixtureSelection": f"{N_CASES} real flagged windows selected across the score range from a {POOL_CAP}-window test pool. Selection does not affect the benchmark metrics.",
    "caveats": [
        "Public IBM AML data; labels are the dataset’s, not a bank’s ground truth.",
        "The kernel runs on a simulator for the benchmark and on real hardware for the degradation report; no quantum advantage is claimed.",
        "Near-zero net flow and near-threshold transfers also occur in legitimate businesses.",
    ],
    "sources": [
        {"title": "IBM Transactions for Anti-Money-Laundering (HI-Small)", "url": "Kaggle: ealtman2019"},
        {"title": "FD4QC — classical vs quantum AML", "url": "https://arxiv.org/abs/2507.19402"},
    ],
}

model = {
    "supportVectorCount": int((np.abs(svc.dual_coef_) > 0).sum()),
    "trainingIds": [w.replace("|", "-") for w in tr_wids],
    "fixedFeatureTransforms": "MinMax to [0, π] per feature (angle encoding).",
    "svc": {"C": 16, "classWeight": "balanced", "kernel": "precomputed"},
    "reproducibility": "q-sieve-demo/make_real_demo.py regenerates this from data/X.csv + data/HI-Small_Trans.csv.",
}

out = {"meta": meta, "features": FEATURES, "policy": {
    "clearMax": CLEAR_MAX, "escalateMin": ESCALATE_MIN, "recallTarget": 0.90,
    "thresholdBasis": "Illustrative demo route cutoffs (clear < 0.20, priority ≥ 0.75).",
    "closureRule": "Only clear-eligible windows may be auto-cleared; escalated or retained windows stay open.",
    "autoClearEnabledByDefault": False,
}, "evaluation": evaluation, "cases": cases, "model": model}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
print(f"\nwrote {OUT}")
print(f"  cases: {len(cases)}  (clear/review/escalate = "
      f"{sum(c['status']=='clear' for c in cases)}/"
      f"{sum(c['status']=='review' for c in cases)}/"
      f"{sum(c['status']=='escalate' for c in cases)})")
print(f"  demo-cutoff 0.20 on pool: {n_below} below, {n_susp_below} suspicious "
      f"(coverage {cov:.1%}, error {err:.1%})")
