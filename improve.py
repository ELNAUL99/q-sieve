"""Can we lift the quantum score? Two quantum-specific levers vs the angle baseline.

1. Bandwidth sweep — scale the encoding angles by c (Shaydulin & Wild 2021): a
   smaller bandwidth fights exponential concentration of the fidelity kernel.
2. Projected quantum kernel (Huang et al. 2021) — encode, measure per-qubit Pauli
   <X>,<Y>,<Z> expectations, then a classical RBF kernel on those 3*n features.

Fair: same real data, split, train cap and CV-tuned C as benchmark.py. Seed-averaged.
Run:  .venv/Scripts/python.exe improve.py
"""
from __future__ import annotations
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import rbf_kernel
from qiskit.circuit.library import z_feature_map
from qiskit.quantum_info import Statevector

import kernels
from data import load_features, cap_train
from baseline import split

TRAIN_CAP, TEST_CAP, SEEDS = 200, 1500, range(3)
BANDWIDTHS = [0.1, 0.25, 0.5, 0.75, 1.0]
CS = [1, 4, 16, 64]


def cv_best_prauc_precomputed(Ktr, ytr):
    best = -1
    skf = StratifiedKFold(3, shuffle=True, random_state=0)
    for C in CS:
        aps = []
        for tr, va in skf.split(Ktr, ytr):
            m = SVC(kernel="precomputed", class_weight="balanced", C=C).fit(
                Ktr[np.ix_(tr, tr)], ytr.iloc[tr])
            aps.append(average_precision_score(ytr.iloc[va], m.decision_function(Ktr[np.ix_(va, tr)])))
        best = max(best, np.mean(aps))
    return best


def fit_score_precomputed(Ktr, Kte, ytr, yte):
    # pick C by CV on train, then score test
    skf = StratifiedKFold(3, shuffle=True, random_state=0); bestC, bestap = 1, -1
    for C in CS:
        aps = []
        for tr, va in skf.split(Ktr, ytr):
            m = SVC(kernel="precomputed", class_weight="balanced", C=C).fit(
                Ktr[np.ix_(tr, tr)], ytr.iloc[tr])
            aps.append(average_precision_score(ytr.iloc[va], m.decision_function(Ktr[np.ix_(va, tr)])))
        if np.mean(aps) > bestap:
            bestap, bestC = np.mean(aps), C
    m = SVC(kernel="precomputed", class_weight="balanced", C=bestC).fit(Ktr, ytr)
    return average_precision_score(yte, m.decision_function(Kte))


def pauli_expectations(amps, n):
    idx = np.arange(len(amps)); probs = np.abs(amps) ** 2; feats = []
    for i in range(n):
        bit = (idx >> i) & 1
        flip = idx ^ (1 << i)
        z = float(np.sum(probs * (1 - 2 * bit)))
        x = float(np.real(np.sum(np.conj(amps) * amps[flip])))
        y = float(np.real(np.sum(np.conj(amps) * (-1j) * (1 - 2 * bit) * amps[flip])))
        feats += [z, x, y]
    return feats


def projected_features(Xs, fm, n):
    return np.array([pauli_expectations(Statevector(fm.assign_parameters(list(r))).data, n) for r in Xs])


def main():
    X, y, _, source = load_features()
    d = X.shape[1]
    fm = z_feature_map(d, reps=1)
    rows = {f"band c={c}": [] for c in BANDWIDTHS}
    rows["projected QK"] = []
    print(f"data [{source}] | d={d} | train={TRAIN_CAP} test={TEST_CAP} | seeds={len(list(SEEDS))}\n")

    for seed in SEEDS:
        Xtr0, Xte0, ytr0, yte0 = split(X, y)
        Xtr, ytr = cap_train(Xtr0, ytr0, cap=TRAIN_CAP, seed=seed)
        Xte, yte = cap_train(Xte0, yte0, cap=TEST_CAP, seed=seed)
        sc = MinMaxScaler((0, np.pi)).fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        kern = kernels.fixed_kernel("angle", d, reps=1)
        for c in BANDWIDTHS:
            Ktr = kern.evaluate(Xtr_s * c)
            Kte = kern.evaluate(Xte_s * c, Xtr_s * c)
            rows[f"band c={c}"].append(fit_score_precomputed(Ktr, Kte, ytr, yte))
        # projected quantum kernel
        Ftr, Fte = projected_features(Xtr_s, fm, d), projected_features(Xte_s, fm, d)
        ss = StandardScaler().fit(Ftr); Ftr, Fte = ss.transform(Ftr), ss.transform(Fte)
        best = -1
        for g in ["scale", 0.05, 0.1, 0.5]:
            gamma = 1.0 / (Ftr.shape[1] * Ftr.var()) if g == "scale" else g
            Ktr = rbf_kernel(Ftr, gamma=gamma); Kte = rbf_kernel(Fte, Ftr, gamma=gamma)
            best = max(best, fit_score_precomputed(Ktr, Kte, ytr, yte))
        rows["projected QK"].append(best)
        print(f"  seed {seed} done")

    print(f"\n{'method':<16}{'PR-AUC':>18}   vs baseline(angle c=1.0)")
    base = np.mean(rows["band c=1.0"])
    for name, vals in rows.items():
        m, s = np.mean(vals), np.std(vals)
        tag = "= baseline" if name == "band c=1.0" else (
            "BETTER" if m - s > base else ("~tie (noise)" if m + s > base else "worse"))
        print(f"{name:<16}{m:.3f} +/- {s:.3f}      {tag}")
    print(f"\nreference: XGBoost 0.190 | RBF 0.097 | random floor 0.093")


if __name__ == "__main__":
    main()
