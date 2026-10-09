"""sweep.py — disciplined trial-and-error over quantum-kernel configs.

Every config is run over several seeds and reported as mean +/- std PR-AUC; a
config only "wins" if it clears the noise band. All kernel logic lives in
kernels.py; this file is the grid + the seed-repeated comparison.

Run:  python sweep.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import kernels
from data import load_features, cap_train
from baseline import split, train_baselines, _ranking_score
from metrics import rank_within_noise

TRAIN_CAP = 150
TEST_CAP = 1500
SEEDS = range(5)

CONFIGS = [
    {"name": "ZZ linear r1", "kind": "fixed", "encoding": "zz", "entanglement": "linear", "reps": 1},
    {"name": "ZZ linear r2", "kind": "fixed", "encoding": "zz", "entanglement": "linear", "reps": 2},
    {"name": "ZZ full r1",   "kind": "fixed", "encoding": "zz", "entanglement": "full",   "reps": 1},
    {"name": "ZZ full r2",   "kind": "fixed", "encoding": "zz", "entanglement": "full",   "reps": 2},
    {"name": "Angle r1",     "kind": "fixed", "encoding": "angle", "reps": 1},
    {"name": "ZZ feat-train", "kind": "train_featuremap", "entanglement": "linear", "reps": 1, "maxiter": 25},
    {"name": "ZZ scale-train", "kind": "train_scales", "entanglement": "linear", "maxiter": 40},
    {"name": "Amplitude(sim)", "kind": "amp"},
    {"name": "Amplitude train", "kind": "amp_train", "maxiter": 40},
    {"name": "PCA4 -> Amp",    "kind": "pca_amp", "k": 4},
    {"name": "PCA2 -> Amp",    "kind": "pca_amp", "k": 2},
    {"name": "XGBoost",        "kind": "xgb"},
    {"name": "RBF-SVC",        "kind": "rbf"},
]


def _ap(ytr, Ktr, yte, Kte):
    return average_precision_score(yte, kernels.svc_scores(Ktr, ytr, Kte))


def config_ap(cfg, Xtr, ytr, Xte, yte):
    kind = cfg["kind"]
    d = Xtr.shape[1]

    if kind == "fixed":
        Xtr_s, Xte_s = kernels.to_angles(Xtr, Xte)
        k = kernels.fixed_kernel(cfg["encoding"], d, cfg.get("reps", 1), cfg.get("entanglement", "linear"))
        return _ap(ytr, k.evaluate(Xtr_s), yte, k.evaluate(Xte_s, Xtr_s))

    if kind == "train_featuremap":
        Xtr_s, Xte_s = kernels.to_angles(Xtr, Xte)
        tk = kernels.train_zz_featuremap(Xtr_s, ytr, cfg["reps"], cfg["entanglement"], cfg.get("maxiter", 25))
        return _ap(ytr, tk.evaluate(Xtr_s), yte, tk.evaluate(Xte_s, Xtr_s))

    if kind == "train_scales":
        Xtr_s, Xte_s = kernels.to_angles(Xtr, Xte)
        tk = kernels.train_zz_scales(Xtr_s, ytr, cfg.get("entanglement", "linear"), cfg.get("maxiter", 40))
        return _ap(ytr, tk.evaluate(Xtr_s), yte, tk.evaluate(Xte_s, Xtr_s))

    if kind == "amp":
        Xtr_s, Xte_s = kernels.to_angles(Xtr, Xte, hi=1.0)
        return _ap(ytr, kernels.amplitude_kernel(Xtr_s, Xtr_s),
                   yte, kernels.amplitude_kernel(Xte_s, Xtr_s))

    if kind == "amp_train":
        Xtr_s, Xte_s = kernels.to_angles(Xtr, Xte, hi=1.0)
        th = kernels.train_amplitude_scales(Xtr_s, ytr, cfg.get("maxiter", 40))
        return _ap(ytr, kernels.amplitude_kernel(Xtr_s * th, Xtr_s * th),
                   yte, kernels.amplitude_kernel(Xte_s * th, Xtr_s * th))

    if kind == "pca_amp":
        Ptr, Pte = kernels.pca_reduce(Xtr, Xte, cfg["k"])
        return _ap(ytr, kernels.amplitude_kernel(Ptr, Ptr),
                   yte, kernels.amplitude_kernel(Pte, Ptr))

    model = train_baselines(Xtr, ytr)["xgb" if kind == "xgb" else "rbf"]
    return average_precision_score(yte, _ranking_score(model, Xte))


def run():
    X, y, _, source = load_features()
    Xtr0, Xte0, ytr0, yte0 = split(X, y)
    print(f"data [{source}]: {len(X)} windows | train_cap={TRAIN_CAP} "
          f"test_cap={TEST_CAP} | seeds={len(list(SEEDS))}\n")

    records = []
    for cfg in CONFIGS:
        aps = []
        for seed in SEEDS:
            Xtr, ytr = cap_train(Xtr0, ytr0, cap=TRAIN_CAP, seed=seed)
            Xte, yte = cap_train(Xte0, yte0, cap=TEST_CAP, seed=seed)
            aps.append(config_ap(cfg, Xtr, ytr, Xte, yte))
        records.append({"config": cfg["name"], "mean": np.mean(aps),
                        "std": np.std(aps), "n": len(aps)})
        print(f"  {cfg['name']:<18} {np.mean(aps):.3f} +/- {np.std(aps):.3f}")

    df = rank_within_noise(pd.DataFrame(records), "config")
    print("\n" + "=" * 56)
    print("RANKED (mean PR-AUC over seeds; higher is better)")
    print("=" * 56)
    print(df.to_string(index=False, formatters={"mean": "{:.3f}".format, "std": "{:.3f}".format}))
    print(f"\nrandom-guess baseline (prevalence) ~ {float(yte0.mean()):.3f}")
    df.to_csv("data/sweep_results.csv", index=False)
    print("\nwrote data/sweep_results.csv")


if __name__ == "__main__":
    run()
