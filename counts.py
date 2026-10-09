"""counts.py — turn PR-AUC into head-counts.

PR-AUC is threshold-free; a bank wants numbers. This fixes the 90%-recall
operating point (catch 90% of launderers) and reports, per model, the raw
confusion matrix on the FULL held-out test split:

  found (TP)  — launderers caught
  missed (FN) — launderers let through
  FP          — clean windows wrongly flagged (the analyst's wasted work)
  marked      — TP + FP (everything the model surfaces)
  cleared(TN) — clean windows safely set aside

Then it projects the measured rates onto the full flagged universe
(388,538 windows a velocity rule raises: 4,084 suspicious + 384,454 clean).

Run:  .venv/Scripts/python.exe counts.py
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from data import load_features, cap_train
from baseline import split, _ranking_score
from benchmark import tuned_quantum_angle, tuned_xgb, tuned_rbf, quantum_scores
from metrics import RECALL

TRAIN_CAP = 200          # same cap the quantum kernel is benchmarked at
SEED = 0

# the full flagged universe build_features reports (pre negative-cap)
FLAGGED_POS = 4_084
FLAGGED_NEG = 384_454


def confusion_at_recall(y, score, target=RECALL):
    thr = np.quantile(score[y == 1], 1 - target)   # 90%-recall threshold
    pred = score >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    tn = int(((~pred) & (y == 0)).sum())
    return tp, fp, fn, tn


def main():
    X, y, _, source = load_features()
    Xtr0, Xte0, ytr0, yte0 = split(X, y)
    Xtr, ytr = cap_train(Xtr0, ytr0, cap=TRAIN_CAP, seed=SEED)
    yte = yte0.to_numpy()
    P, N = int(yte.sum()), int((yte == 0).sum())

    print(f"data [{source}] | trained on {len(ytr)} windows "
          f"({int(ytr.sum())} pos) | tested on {len(yte)} windows "
          f"({P} launderers + {N} clean)\n")

    qb, _ = tuned_quantum_angle(Xtr, ytr, SEED)
    xgb, _ = tuned_xgb(Xtr, ytr, SEED)
    rbf, _ = tuned_rbf(Xtr, ytr, SEED)
    scored = {
        "xgboost(tuned)":  _ranking_score(xgb, Xte0),
        "quantum(angle)":  quantum_scores(qb, Xte0),
        "rbf(tuned)":      _ranking_score(rbf, Xte0),
    }

    print(f"operating point: {RECALL:.0%} recall (catch {RECALL:.0%} of launderers)\n")
    hdr = f"{'model':<16}{'PR-AUC':>8}{'found':>7}{'missed':>8}{'FP':>7}{'marked':>8}{'cleared':>9}"
    print(hdr); print("-" * len(hdr))
    scale = FLAGGED_NEG / N
    proj = []
    for name, s in scored.items():
        tp, fp, fn, tn = confusion_at_recall(yte, s)
        ap = average_precision_score(yte, s)
        print(f"{name:<16}{ap:>8.3f}{tp:>7}{fn:>8}{fp:>7}{tp+fp:>8}{tn:>9}")
        proj.append((name, fp, round(fp * scale)))

    print(f"\nprojected onto the full flagged book "
          f"({FLAGGED_POS:,} suspicious + {FLAGGED_NEG:,} clean), at {RECALL:.0%} recall:")
    print(f"  all models catch ~{round(FLAGGED_POS*RECALL):,} launderers, "
          f"miss ~{round(FLAGGED_POS*(1-RECALL)):,}")
    print(f"  {'model':<16}{'clean windows wrongly flagged (FP)':>36}")
    for name, _fp, fp_full in proj:
        print(f"  {name:<16}{fp_full:>30,}")


if __name__ == "__main__":
    main()
