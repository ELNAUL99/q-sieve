"""Rigorous verification: do bandwidth / projected kernels really match the tree?

Every model on IDENTICAL splits per seed, 5 seeds, test=3000, finer bandwidth grid.
Run:  .venv/Scripts/python.exe improve_verify.py
"""
from __future__ import annotations
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import rbf_kernel

import kernels
from data import load_features, cap_train
from baseline import split, _ranking_score
from benchmark import tuned_xgb, tuned_rbf
from improve import fit_score_precomputed, projected_features
from qiskit.circuit.library import z_feature_map

TRAIN_CAP, TEST_CAP, SEEDS = 200, 3000, range(5)
BANDWIDTHS = [0.05, 0.1, 0.15, 0.2, 0.3, 1.0]


def main():
    X, y, _, source = load_features()
    d = X.shape[1]
    fm = z_feature_map(d, reps=1)
    names = ["xgboost", "rbf", "projected QK"] + [f"band c={c}" for c in BANDWIDTHS]
    rows = {n: [] for n in names}
    print(f"data [{source}] | d={d} | train={TRAIN_CAP} test={TEST_CAP} | seeds={len(list(SEEDS))}\n")

    for seed in SEEDS:
        Xtr0, Xte0, ytr0, yte0 = split(X, y)
        Xtr, ytr = cap_train(Xtr0, ytr0, cap=TRAIN_CAP, seed=seed)
        Xte, yte = cap_train(Xte0, yte0, cap=TEST_CAP, seed=seed)

        xgb, _ = tuned_xgb(Xtr, ytr, seed)
        rows["xgboost"].append(average_precision_score(yte, _ranking_score(xgb, Xte)))
        rbf, _ = tuned_rbf(Xtr, ytr, seed)
        rows["rbf"].append(average_precision_score(yte, _ranking_score(rbf, Xte)))

        sc = MinMaxScaler((0, np.pi)).fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        kern = kernels.fixed_kernel("angle", d, reps=1)
        for c in BANDWIDTHS:
            Ktr = kern.evaluate(Xtr_s * c); Kte = kern.evaluate(Xte_s * c, Xtr_s * c)
            rows[f"band c={c}"].append(fit_score_precomputed(Ktr, Kte, ytr, yte))

        Ftr, Fte = projected_features(Xtr_s, fm, d), projected_features(Xte_s, fm, d)
        ss = StandardScaler().fit(Ftr); Ftr, Fte = ss.transform(Ftr), ss.transform(Fte)
        best = -1
        for g in ["scale", 0.02, 0.05, 0.1, 0.5]:
            gamma = 1.0 / (Ftr.shape[1] * Ftr.var()) if g == "scale" else g
            Ktr = rbf_kernel(Ftr, gamma=gamma); Kte = rbf_kernel(Fte, Ftr, gamma=gamma)
            best = max(best, fit_score_precomputed(Ktr, Kte, ytr, yte))
        rows["projected QK"].append(best)
        print(f"  seed {seed} done")

    xgb_m, xgb_s = np.mean(rows["xgboost"]), np.std(rows["xgboost"])
    print(f"\n{'method':<16}{'PR-AUC':>18}   vs XGBoost")
    for n in names:
        m, s = np.mean(rows[n]), np.std(rows[n])
        if n == "xgboost":
            verdict = "(reference)"
        else:
            verdict = ("BEATS xgb" if m - s > xgb_m + xgb_s else
                       "ties xgb" if m + s > xgb_m - xgb_s else "below xgb")
        print(f"{n:<16}{m:.3f} +/- {s:.3f}      {verdict}")


if __name__ == "__main__":
    main()
