"""Layer 2 — Quantum kernel classifier (simulator).

The quantum computer answers one question: how similar are two windows? Each
feature-row is encoded into a circuit (one qubit per feature) and the fidelity
(overlap) between rows forms the kernel matrix; a classical SVC draws the
boundary from it.

The kernel builders (fixed / trained / amplitude) live in kernels.py. This file
holds only the scikit-learn-compatible classifier that wraps one.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import kernels

RANDOM_STATE = 0


def build_kernel(n_features: int, reps: int = 1, entanglement: str = "linear"):
    """ZZFeatureMap fidelity kernel — thin wrapper over kernels.fixed_kernel."""
    return kernels.fixed_kernel("zz", n_features, reps=reps, entanglement=entanglement)


class QuantumKernelClassifier(BaseEstimator, ClassifierMixin):
    """A scikit-learn classifier backed by the quantum fidelity kernel.

    Takes RAW feature rows (scales to [0, pi] internally) so it matches the
    classical baseline's interface. Also exposes .similarity() for precedent
    retrieval in the trust layer.
    """

    def __init__(self, reps: int = 1, entanglement: str = "linear",
                 class_weight="balanced"):
        self.reps = reps
        self.entanglement = entanglement
        self.class_weight = class_weight

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.scaler_ = MinMaxScaler((0, np.pi)).fit(X)
        self.X_train_ = self.scaler_.transform(X)
        self.y_train_ = y
        self.kernel_ = build_kernel(X.shape[1], self.reps, self.entanglement)
        Ktr = self.kernel_.evaluate(self.X_train_)
        self.svc_ = SVC(kernel="precomputed", class_weight=self.class_weight,
                        random_state=RANDOM_STATE).fit(Ktr, y)
        return self

    def similarity(self, X):
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        return self.kernel_.evaluate(Xs, self.X_train_)

    def decision_function(self, X):
        return self.svc_.decision_function(self.similarity(X))

    def predict(self, X):
        return self.svc_.predict(self.similarity(X))


if __name__ == "__main__":
    import time
    from sklearn.metrics import average_precision_score
    from demo_data import make_demo_queue
    from baseline import split, train_baselines, evaluate
    from data import cap_train

    X, y, _, _ = make_demo_queue(n_each=40)
    Xtr, Xte, ytr, yte = split(X, y)
    Xtr, ytr = cap_train(Xtr, ytr, cap=80)
    Xte, yte = cap_train(Xte, yte, cap=30)

    t0 = time.time()
    qk = QuantumKernelClassifier().fit(Xtr, ytr)
    q_ap = average_precision_score(yte, qk.decision_function(Xte))
    print(f"quantum: train={len(Xtr)} test={len(Xte)} qubits={X.shape[1]} | "
          f"PR-AUC {q_ap:.3f} ({time.time()-t0:.1f}s)")
    for name, ap in evaluate(train_baselines(Xtr, ytr), Xte, yte).items():
        print(f"classical {name} PR-AUC: {ap:.3f}")
