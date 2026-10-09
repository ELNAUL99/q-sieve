"""Export a minimal, frozen hardware batch (runs in the ML venv; no quota).

Trains the angle-encoding kernel SVC on the simulator, then exports everything a
hardware run needs: support-vector angle rows + SVC dual + intercept, a small
test set, and the exact (simulator) PR-AUC baseline. Small (few qubits, few
circuits) to fit a tight hardware budget.

Run:  .venv/Scripts/python.exe export_hw_batch.py   -> data/hw_batch.json
"""

import json

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score

import kernels
from data import load_features, cap_train
from baseline import split

D_HW = 6        # qubits on hardware (shallow angle circuit; low readout noise)
N_TRAIN = 60    # small train -> few support vectors -> few circuits
N_TEST = 16
OUT = "data/hw_batch.json"


def main():
    X, y, _, source = load_features()
    Xtr, Xte, ytr, yte = split(X, y)
    Xtr, ytr = cap_train(Xtr, ytr, cap=N_TRAIN, seed=0)
    Xte, yte = cap_train(Xte, yte, cap=N_TEST, seed=0)

    sel = SelectKBest(f_classif, k=D_HW).fit(Xtr, ytr)
    scaler = MinMaxScaler((0, np.pi)).fit(sel.transform(Xtr))
    Atr = np.clip(scaler.transform(sel.transform(Xtr)), 0, np.pi)
    Ate = np.clip(scaler.transform(sel.transform(Xte)), 0, np.pi)

    kernel = kernels.fixed_kernel("angle", D_HW, reps=1)
    svc = SVC(kernel="precomputed", class_weight="balanced").fit(kernel.evaluate(Atr), ytr.values)

    sv = svc.support_
    exact_dec = svc.decision_function(kernel.evaluate(Ate, Atr))
    exact_prauc = float(average_precision_score(yte, exact_dec))

    batch = {
        "d": D_HW, "encoding": "angle", "source": source,
        "sv_angles": Atr[sv].tolist(),
        "dual": svc.dual_coef_[0].tolist(),
        "intercept": float(svc.intercept_[0]),
        "test_angles": Ate.tolist(),
        "test_labels": yte.astype(int).tolist(),
        "exact_decision": exact_dec.tolist(),
        "exact_prauc": exact_prauc,
    }
    json.dump(batch, open(OUT, "w"))
    n_circ = len(sv) * len(Ate)
    print(f"source={source} | qubits={D_HW} | support vectors={len(sv)} | test={len(Ate)}")
    print(f"hardware circuits needed: {len(sv)} x {len(Ate)} = {n_circ}")
    print(f"exact (simulator) PR-AUC: {exact_prauc:.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
