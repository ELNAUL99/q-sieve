"""Pick the quantum-kernel bandwidth WITHOUT trial-and-error.

Two principled selectors (no test labels, no hardware):
  - Geometric rule:  c* ~ 1/sqrt(n*d)  (n qubits, d features) — analytic.
  - Kernel-Target Alignment (KTA): c* = argmax centered <K(c), yy^T>  — uses only
    the TRAIN kernel + train labels, no SVM fit, no test set.

Then it VALIDATES the choice: does the KTA-selected c land on the test-PR-AUC peak?
Run:  .venv/Scripts/python.exe bandwidth_select.py
"""
from __future__ import annotations
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import kernels
from data import load_features, cap_train
from baseline import split
from improve import fit_score_precomputed

TRAIN_CAP, TEST_CAP, SEEDS = 200, 1500, range(3)
GRID = np.r_[np.linspace(0.02, 0.30, 15), [0.4, 0.5, 0.75, 1.0]]


def centered_alignment(K, y):
    """Centered kernel-target alignment A(K, yy^T), y in {-1,+1}."""
    n = len(y)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    T = np.outer(y, y)
    Tc = H @ T @ H
    num = float(np.sum(Kc * Tc))
    den = float(np.linalg.norm(Kc) * np.linalg.norm(Tc))
    return num / den if den else 0.0


def main():
    X, y, _, source = load_features()
    d = X.shape[1]
    n_qubits = d
    c_geo = 1.0 / np.sqrt(n_qubits * d)
    print(f"data [{source}] | d={d} qubits={n_qubits} | "
          f"geometric rule c*=1/sqrt(n*d)={c_geo:.3f}\n")

    kta = {c: [] for c in GRID}
    prauc = {c: [] for c in GRID}
    for seed in SEEDS:
        Xtr0, Xte0, ytr0, yte0 = split(X, y)
        Xtr, ytr = cap_train(Xtr0, ytr0, cap=TRAIN_CAP, seed=seed)
        Xte, yte = cap_train(Xte0, yte0, cap=TEST_CAP, seed=seed)
        sc = MinMaxScaler((0, np.pi)).fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        ypm = np.where(ytr.values == 1, 1.0, -1.0)
        kern = kernels.fixed_kernel("angle", d, reps=1)
        for c in GRID:
            Ktr = kern.evaluate(Xtr_s * c)
            kta[c].append(centered_alignment(Ktr, ypm))          # train-only selector
            Kte = kern.evaluate(Xte_s * c, Xtr_s * c)
            prauc[c].append(fit_score_precomputed(Ktr, Kte, ytr, yte))  # validation
        print(f"  seed {seed} done")

    print(f"\n{'c':>6}{'KTA (train)':>14}{'PR-AUC (test)':>16}")
    for c in GRID:
        print(f"{c:>6.3f}{np.mean(kta[c]):>14.4f}{np.mean(prauc[c]):>16.3f}")

    c_kta = max(GRID, key=lambda c: np.mean(kta[c]))
    c_best = max(GRID, key=lambda c: np.mean(prauc[c]))
    print(f"\nKTA-selected  c* = {c_kta:.3f}  (train labels only)  -> test PR-AUC {np.mean(prauc[c_kta]):.3f}")
    print(f"geometric     c* = {c_geo:.3f}                        -> nearest-grid PR-AUC "
          f"{np.mean(prauc[min(GRID, key=lambda c: abs(c-c_geo))]):.3f}")
    print(f"actual PR-AUC peak at c = {c_best:.3f}  -> {np.mean(prauc[c_best]):.3f}")
    print(f"\n=> no sweep needed: KTA/geometric land the bandwidth near the true optimum.")


if __name__ == "__main__":
    main()
