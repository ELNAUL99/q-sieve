"""Classic-first, quantum-enhance. A separate cascade — nothing else imports this.

XGBoost scores every flagged window. Windows whose classical P(suspicious) sits
near the decision boundary are the ambiguous ones; only those are re-scored by
the angle QSVM. Confident classical decisions stay as they are.

Same data contract, split, and caps as the benchmark, so the two can be compared
without changing benchmark.py. Run:  python hybrid.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import kernels
from baseline import split, train_baselines
from data import cap_train, load_features
from metrics import fp_eliminated_at_recall

TRAIN_CAP = 200
TEST_CAP = 3000
# Classical P(suspicious) inside [0.5 - BAND, 0.5 + BAND] is ambiguous.
BAND = 0.15


def classic_first(Xtr, ytr):
    """Fit the fast classical stage on the training windows."""
    return train_baselines(Xtr, ytr)["xgb"]


def ambiguous_mask(p_suspicious: np.ndarray, band: float = BAND) -> np.ndarray:
    """True where the classical model is too close to 0.5 to trust."""
    p = np.asarray(p_suspicious, dtype=float)
    return (p >= 0.5 - band) & (p <= 0.5 + band)


def quantum_enhance(Xtr, ytr, X_pass: pd.DataFrame) -> np.ndarray:
    """Angle QSVM scores for the windows the classical stage passed on.

    The kernel is fit on the training windows (same cap as the benchmark).
    Only `X_pass` is evaluated — that is the refinement set.
    """
    sc = MinMaxScaler((0, np.pi)).fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    kernel = kernels.fixed_kernel("angle", Xtr.shape[1], reps=1)
    model = SVC(kernel="precomputed", class_weight="balanced").fit(
        kernel.evaluate(Xtr_s), ytr)
    if len(X_pass) == 0:
        return np.array([])
    return model.decision_function(kernel.evaluate(sc.transform(X_pass), Xtr_s))


def classic_first_quantum_enhance(Xtr, ytr, Xte, yte, band: float = BAND) -> dict:
    """Score the whole queue classically, then let the QSVM reorder the ambiguous slice.

    Quantum margins are placed back inside the classical probability band, so
    confident windows keep their classical score and only the passed rows move.
    """
    classical = classic_first(Xtr, ytr)
    p = classical.predict_proba(Xte)[:, 1]
    mask = ambiguous_mask(p, band)

    q = quantum_enhance(Xtr, ytr, Xte.loc[mask])
    hybrid = p.copy()
    if mask.any():
        span = q.max() - q.min()
        q01 = (q - q.min()) / span if span > 0 else np.full(q.shape, 0.5)
        lo, hi = 0.5 - band, 0.5 + band
        hybrid[mask] = lo + q01 * (hi - lo)

    y = np.asarray(yte)
    return {
        "n": int(len(Xte)),
        "n_passed": int(mask.sum()),
        "n_passed_suspicious": int(y[mask].sum()) if mask.any() else 0,
        "pr_auc_classical": float(average_precision_score(y, p)),
        "pr_auc_hybrid": float(average_precision_score(y, hybrid)),
        "fp_elim_classical": float(fp_eliminated_at_recall(y, p)),
        "fp_elim_hybrid": float(fp_eliminated_at_recall(y, hybrid)),
        "p_classical": p,
        "p_hybrid": hybrid,
        "passed": mask,
    }


def main():
    X, y, _, source = load_features()
    Xtr0, Xte0, ytr0, yte0 = split(X, y)
    Xtr, ytr = cap_train(Xtr0, ytr0, cap=TRAIN_CAP, seed=0)
    Xte, yte = cap_train(Xte0, yte0, cap=TEST_CAP, seed=0)
    r = classic_first_quantum_enhance(Xtr, ytr, Xte, yte)

    print(f"data [{source}] | train={len(Xtr)} test={r['n']} | band=+/-{BAND}")
    print(f"classical scored all {r['n']}")
    print(f"passed to quantum: {r['n_passed']} "
          f"({r['n_passed_suspicious']} of them suspicious)")
    print(f"PR-AUC   classical {r['pr_auc_classical']:.3f}  "
          f"hybrid {r['pr_auc_hybrid']:.3f}")
    print(f"FP-elim  classical {r['fp_elim_classical']:.1%}  "
          f"hybrid {r['fp_elim_hybrid']:.1%}")


if __name__ == "__main__":
    main()
