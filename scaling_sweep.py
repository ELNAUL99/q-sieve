"""Scaling sweep for the quantum (angle) kernel on real AML data.

Features become rotation angles, so a heavy-tailed feature linearly scaled puts
99% of samples in a tiny slice of [0, pi]. Distribution-shaping transforms (log,
quantile) spread samples out. Tests the untested shape transforms.

Run:  python scaling_sweep.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import (MinMaxScaler, StandardScaler, RobustScaler,
                                    QuantileTransformer, PowerTransformer)
from sklearn.metrics import average_precision_score

import kernels
from data import load_features, cap_train
from baseline import split
from metrics import rank_within_noise

TRAIN_CAP = 200
TEST_CAP = 2000
SEEDS = range(5)


def _signed_log1p(X):
    return np.sign(X) * np.log1p(np.abs(X))


def make_angles(name, Xtr, Xte):
    hi = np.pi
    if name == "MinMax[0,pi]":
        shaper = None; hi = np.pi
    elif name == "MinMax[0,pi/2]":
        shaper = None; hi = np.pi / 2
    elif name == "MinMax[0,2pi]":
        shaper = None; hi = 2 * np.pi
    elif name == "Standard->[0,pi]":
        shaper = StandardScaler()
    elif name == "Robust->[0,pi]":
        shaper = RobustScaler()
    elif name == "log1p->[0,pi]":
        shaper = "log"
    elif name == "Quantile->[0,pi]":
        shaper = QuantileTransformer(output_distribution="uniform",
                                     n_quantiles=min(TRAIN_CAP, len(Xtr)), random_state=0)
    elif name == "Power(YJ)->[0,pi]":
        shaper = PowerTransformer(method="yeo-johnson")
    else:
        raise ValueError(name)

    tr, te = np.asarray(Xtr, float), np.asarray(Xte, float)
    if shaper == "log":
        tr, te = _signed_log1p(tr), _signed_log1p(te)
    elif shaper is not None:
        shaper.fit(tr); tr, te = shaper.transform(tr), shaper.transform(te)

    mm = MinMaxScaler((0, hi)).fit(tr)
    return np.clip(mm.transform(tr), 0, hi), np.clip(mm.transform(te), 0, hi)


def angle_ap(Xtr_a, ytr, Xte_a, yte):
    k = kernels.fixed_kernel("angle", Xtr_a.shape[1], reps=1)
    return average_precision_score(
        yte, kernels.svc_scores(k.evaluate(Xtr_a), ytr, k.evaluate(Xte_a, Xtr_a)))


def main():
    X, y, _, source = load_features()
    Xtr0, Xte0, ytr0, yte0 = split(X, y)
    print(f"data [{source}] | features={X.shape[1]} | train_cap={TRAIN_CAP} "
          f"test_cap={TEST_CAP} | seeds={len(list(SEEDS))}\n")

    configs = ["MinMax[0,pi]", "MinMax[0,pi/2]", "MinMax[0,2pi]", "Standard->[0,pi]",
               "Robust->[0,pi]", "log1p->[0,pi]", "Quantile->[0,pi]", "Power(YJ)->[0,pi]"]
    records = []
    for name in configs:
        aps = []
        for seed in SEEDS:
            Xtr, ytr = cap_train(Xtr0, ytr0, cap=TRAIN_CAP, seed=seed)
            Xte, yte = cap_train(Xte0, yte0, cap=TEST_CAP, seed=seed)
            Xtr_a, Xte_a = make_angles(name, Xtr, Xte)
            aps.append(angle_ap(Xtr_a, ytr, Xte_a, yte))
        records.append({"scaling": name, "mean": np.mean(aps), "std": np.std(aps)})
        print(f"  {name:<18} {np.mean(aps):.3f} +/- {np.std(aps):.3f}")

    df = rank_within_noise(pd.DataFrame(records), "scaling")
    print("\n" + "=" * 56)
    print("RANKED — quantum angle kernel by scaling (mean PR-AUC)")
    print("=" * 56)
    print(df.to_string(index=False, formatters={"mean": "{:.3f}".format, "std": "{:.3f}".format}))
    print(f"\nrandom baseline ~ {yte0.mean():.3f}")
    df.to_csv("data/scaling_results.csv", index=False)
    print("wrote data/scaling_results.csv")


if __name__ == "__main__":
    main()
