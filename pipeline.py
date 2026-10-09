"""Unified Q-SIEVE pipeline.

Ties the layers together behind one function so the dashboard and experiments run
the same code path. The classifier is swappable via backend="classical" |
"quantum"; everything downstream is identical.
"""

from __future__ import annotations

import numpy as np

from sklearn.metrics import average_precision_score

from baseline import split, train_baselines
from data import cap_train
from trust import calibrate, auto_clear_curve, choose_threshold

QUANTUM_TRAIN_CAP = 300  # kernel cost grows with n^2 — keep the sim training small


def build_model(X, y, backend: str = "classical", max_error: float = 0.01,
                calibration: str = "isotonic"):
    """Train + calibrate a model, then pick a conservative auto-clear threshold."""
    Xtr, Xte, ytr, yte = split(X, y)
    Xfit, Xcal, yfit, ycal = split(Xtr, ytr)

    if backend == "quantum":
        from quantum_kernel import QuantumKernelClassifier
        Xq, yq = cap_train(Xfit, yfit, cap=QUANTUM_TRAIN_CAP)
        base = QuantumKernelClassifier().fit(Xq, yq)
    elif backend == "classical":
        base = train_baselines(Xfit, yfit)["xgb"]
    else:
        raise ValueError(f"unknown backend: {backend!r}")

    model = calibrate(base, Xcal, ycal, method=calibration)
    curve = auto_clear_curve(model, Xte, yte)
    pick = choose_threshold(curve, max_error=max_error)
    pr_auc = float(average_precision_score(yte, model.predict_proba(Xte)[:, 1]))

    return {
        "model": model, "base": base, "pr_auc": pr_auc,
        "Xfit": Xfit, "yfit": yfit, "Xtr": Xtr, "Xte": Xte,
        "ytr": ytr, "yte": yte, "curve": curve, "pick": pick, "backend": backend,
    }


if __name__ == "__main__":
    from data import load_features

    X, y, _, source = load_features()
    print(f"data source: {source} | X={X.shape} | positives={int(y.sum())}/{len(y)}")
    for backend in ("classical", "quantum"):
        r = build_model(X, y, backend=backend)
        print(f"{backend:>9}: PR-AUC {r['pr_auc']:.3f} | "
              f"auto-clear {r['pick']['cleared_share']:.0%} "
              f"at {r['pick']['error_rate']:.1%} error")
