"""Fair encoding comparison — each encoding swept across bandwidth.

We picked angle at bandwidth c=1, then found bandwidth matters. So: does a
bandwidth-tuned ZZ match or beat a bandwidth-tuned angle? Same splits, seed-avg.
(Amplitude is scale-invariant — bandwidth doesn't apply — so it's excluded.)

Run:  .venv/Scripts/python.exe encoding_bandwidth.py
"""
from __future__ import annotations
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import average_precision_score

import kernels
from data import load_features, cap_train
from baseline import split, _ranking_score
from benchmark import tuned_xgb, tuned_rbf
from improve import fit_score_precomputed

TRAIN_CAP, TEST_CAP, SEEDS = 200, 1500, range(3)
BANDWIDTHS = [0.05, 0.1, 0.2, 0.5, 1.0]
ENCODINGS = ["angle", "zz"]


def main():
    X, y, _, source = load_features()
    d = X.shape[1]
    rows = {f"{enc} c={c}": [] for enc in ENCODINGS for c in BANDWIDTHS}
    rows["xgboost"], rows["rbf"] = [], []
    print(f"data [{source}] | d={d} | train={TRAIN_CAP} test={TEST_CAP} | seeds={len(list(SEEDS))}\n")

    for seed in SEEDS:
        Xtr0, Xte0, ytr0, yte0 = split(X, y)
        Xtr, ytr = cap_train(Xtr0, ytr0, cap=TRAIN_CAP, seed=seed)
        Xte, yte = cap_train(Xte0, yte0, cap=TEST_CAP, seed=seed)
        xgb, _ = tuned_xgb(Xtr, ytr, seed); rows["xgboost"].append(average_precision_score(yte, _ranking_score(xgb, Xte)))
        rbf, _ = tuned_rbf(Xtr, ytr, seed); rows["rbf"].append(average_precision_score(yte, _ranking_score(rbf, Xte)))
        sc = MinMaxScaler((0, np.pi)).fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        for enc in ENCODINGS:
            kern = kernels.fixed_kernel(enc, d, reps=1)
            for c in BANDWIDTHS:
                Ktr = kern.evaluate(Xtr_s * c); Kte = kern.evaluate(Xte_s * c, Xtr_s * c)
                rows[f"{enc} c={c}"].append(fit_score_precomputed(Ktr, Kte, ytr, yte))
        print(f"  seed {seed} done")

    xgb_m, xgb_s = np.mean(rows["xgboost"]), np.std(rows["xgboost"])
    print(f"\n{'method':<14}{'PR-AUC':>18}   vs XGBoost")
    for n, vals in rows.items():
        m, s = np.mean(vals), np.std(vals)
        v = "(reference)" if n == "xgboost" else (
            "BEATS" if m - s > xgb_m + xgb_s else "ties" if m + s > xgb_m - xgb_s else "below")
        print(f"{n:<14}{m:.3f} +/- {s:.3f}      {v}")
    print(f"\nrandom floor ~ 0.093")


if __name__ == "__main__":
    main()
