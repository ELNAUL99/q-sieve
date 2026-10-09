"""Our honest FPR at matched operating points — so FD4QC's 1.36% is comparable.

FPR is a function of the threshold. This reports our FPR at (a) 90% recall,
(b) a precision-matched point (~77%, like FD4QC's QSVM), and (c) the best
precision we can reach. XGBoost and bandwidth-tuned quantum, real test.
Run:  .venv/Scripts/python.exe fpr_tradeoff.py
"""
from __future__ import annotations
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

import kernels
from data import load_features, cap_train
from baseline import split

TRAIN_CAP, TEST_CAP, SEED = 200, 3000, 0


def curve(y, score):
    order = np.argsort(score)[::-1]
    y = np.asarray(y)[order].astype(float)
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    P, N = y.sum(), (1 - y).sum()
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(P, 1)
    fpr = fp / max(N, 1)
    return prec, rec, fpr


def report(name, y, score):
    prec, rec, fpr = curve(y, score)
    i90 = np.argmin(np.abs(rec - 0.90))
    i77 = np.argmin(np.abs(prec - 0.77))
    # best precision at a usable recall (>=5%)
    ok = rec >= 0.05
    ibp = np.argmax(np.where(ok, prec, -1))
    print(f"\n{name}")
    print(f"  @ 90% recall   : FPR {fpr[i90]*100:5.1f}%  precision {prec[i90]*100:4.1f}%  recall {rec[i90]*100:4.1f}%")
    print(f"  @ ~77% prec    : FPR {fpr[i77]*100:5.1f}%  precision {prec[i77]*100:4.1f}%  recall {rec[i77]*100:4.1f}%")
    print(f"  @ best precision: FPR {fpr[ibp]*100:5.1f}%  precision {prec[ibp]*100:4.1f}%  recall {rec[ibp]*100:4.1f}%")


def main():
    X, y, _, _ = load_features()
    Xtr0, Xte0, ytr0, yte0 = split(X, y)
    Xtr, ytr = cap_train(Xtr0, ytr0, cap=TRAIN_CAP, seed=SEED)
    Xte, yte = cap_train(Xte0, yte0, cap=TEST_CAP, seed=SEED)
    print(f"test={len(yte)} windows ({int(yte.sum())} suspicious, {int((yte==0).sum())} clean) | prevalence {yte.mean():.1%}")

    xgb = XGBClassifier(n_estimators=300, max_depth=3,
                        scale_pos_weight=(ytr == 0).sum() / max((ytr == 1).sum(), 1),
                        random_state=0).fit(Xtr, ytr)
    report("XGBoost (tuned classical)", yte, xgb.predict_proba(Xte)[:, 1])

    sc = MinMaxScaler((0, np.pi)).fit(Xtr)
    c = 0.1
    k = kernels.fixed_kernel("angle", Xtr.shape[1], reps=1)
    Ktr = k.evaluate(sc.transform(Xtr) * c)
    svc = SVC(kernel="precomputed", class_weight="balanced", C=16).fit(Ktr, ytr)
    dec = svc.decision_function(k.evaluate(sc.transform(Xte) * c, sc.transform(Xtr) * c))
    report("Quantum angle, bandwidth-tuned (c=0.1)", yte, dec)


if __name__ == "__main__":
    main()
