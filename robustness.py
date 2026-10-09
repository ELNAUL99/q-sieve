"""Robustness of the conclusion to the prevalence (NEG_CAP) choice.

The 9.3% prevalence came from capping negatives at 40k — a tractability knob. This
re-runs the quantum-vs-classical comparison at several prevalences (by
sub-sampling negatives) to show the RANKING is unchanged: the floor moves, the
winner does not. Defends the NEG_CAP choice to the jury.

Run:  python robustness.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import kernels
from data import load_features, cap_train
from baseline import split, train_baselines, _ranking_score

PREVALENCES = [0.09, 0.15, 0.22, 0.33]   # achievable by sub-sampling the 40k negatives
TRAIN_CAP = 150
TEST_CAP = 2000
SEEDS = range(3)


def at_prevalence(X, y, p, seed):
    """Keep all positives + enough negatives to hit prevalence p."""
    rng = np.random.default_rng(seed)
    pos = np.where(y.values == 1)[0]
    neg = np.where(y.values == 0)[0]
    n_neg = int(round(len(pos) * (1 - p) / p))
    if n_neg < len(neg):
        neg = rng.choice(neg, size=n_neg, replace=False)
    keep = np.sort(np.concatenate([pos, neg]))
    return X.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)


def quantum_angle_ap(Xtr, ytr, Xte, yte):
    Xtr_s, Xte_s = kernels.to_angles(Xtr, Xte)
    k = kernels.fixed_kernel("angle", Xtr.shape[1], reps=1)
    return average_precision_score(yte, kernels.svc_scores(k.evaluate(Xtr_s), ytr, k.evaluate(Xte_s, Xtr_s)))


def classical_ap(kind, Xtr, ytr, Xte, yte):
    m = train_baselines(Xtr, ytr)[kind]
    return average_precision_score(yte, _ranking_score(m, Xte))


def main():
    X, y, _, source = load_features()
    print(f"data [{source}] | base prevalence {y.mean():.1%}\n")
    print(f"{'prevalence':>10}{'floor':>8}{'quantum':>9}{'xgb':>8}{'rbf':>8}{'winner':>9}")
    rows = []
    for p in PREVALENCES:
        q, xg, rb = [], [], []
        for seed in SEEDS:
            Xp, yp = at_prevalence(X, y, p, seed)
            Xtr, Xte, ytr, yte = split(Xp, yp)
            Xtr, ytr = cap_train(Xtr, ytr, cap=TRAIN_CAP, seed=seed)
            Xte, yte = cap_train(Xte, yte, cap=TEST_CAP, seed=seed)
            q.append(quantum_angle_ap(Xtr, ytr, Xte, yte))
            xg.append(classical_ap("xgb", Xtr, ytr, Xte, yte))
            rb.append(classical_ap("rbf", Xtr, ytr, Xte, yte))
        mq, mx, mr = np.mean(q), np.mean(xg), np.mean(rb)
        winner = max([("quantum", mq), ("xgb", mx), ("rbf", mr)], key=lambda t: t[1])[0]
        floor = float(yte.mean())
        print(f"{p:>10.0%}{floor:>8.3f}{mq:>9.3f}{mx:>8.3f}{mr:>8.3f}{winner:>9}")
        rows.append({"prevalence": p, "floor": floor, "quantum": mq, "xgb": mx, "rbf": mr, "winner": winner})
    pd.DataFrame(rows).to_csv("data/robustness_results.csv", index=False)
    print("\nIf 'winner' is the same at every prevalence, the NEG_CAP choice does not "
          "change the conclusion. wrote data/robustness_results.csv")


if __name__ == "__main__":
    main()
