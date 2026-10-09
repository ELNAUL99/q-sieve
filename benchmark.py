"""Fair benchmark: quantum kernel vs genuinely-tuned classical kernels.

  1. False positives eliminated at fixed recall (the operative metric)
  2. Uncertain cases — the auto-clear split (what's cleared vs retained)
  3. Fair benchmark — XGBoost and RBF are GridSearch-tuned, identical features /
     splits / training size for every model.

Reported as mean +/- std over seeds. Run:  python benchmark.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

import kernels
from data import load_features, cap_train
from baseline import split, _ranking_score
from metrics import fp_eliminated_at_recall, auto_clear_split, RECALL

TRAIN_CAP = 200
TEST_CAP = 3000
SEEDS = range(3)
SAFE_ERR = 0.01


def tuned_xgb(Xtr, ytr, seed):
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    grid = {"max_depth": [2, 3, 4], "learning_rate": [0.05, 0.1],
            "n_estimators": [150, 300], "min_child_weight": [1, 5]}
    base = XGBClassifier(scale_pos_weight=spw, eval_metric="aucpr", random_state=seed)
    gs = GridSearchCV(base, grid, scoring="average_precision",
                      cv=StratifiedKFold(3, shuffle=True, random_state=seed), n_jobs=-1)
    gs.fit(Xtr, ytr)
    return gs.best_estimator_, gs.best_params_


def tuned_rbf(Xtr, ytr, seed):
    grid = {"C": [1, 4, 16, 64], "gamma": ["scale", 0.1, 1.0]}
    base = SVC(kernel="rbf", class_weight="balanced", random_state=seed)
    gs = GridSearchCV(base, grid, scoring="average_precision",
                      cv=StratifiedKFold(3, shuffle=True, random_state=seed), n_jobs=-1)
    gs.fit(Xtr, ytr)
    return gs.best_estimator_, gs.best_params_


def tuned_quantum_angle(Xtr, ytr, seed):
    sc = MinMaxScaler((0, np.pi)).fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    kernel = kernels.fixed_kernel("angle", Xtr.shape[1], reps=1)
    Ktr = kernel.evaluate(Xtr_s)
    best_C, best_ap = 1.0, -1
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    for C in [1, 4, 16, 64]:
        aps = []
        for tr, va in skf.split(Ktr, ytr):
            m = SVC(kernel="precomputed", class_weight="balanced", C=C).fit(
                Ktr[np.ix_(tr, tr)], ytr.iloc[tr])
            aps.append(average_precision_score(
                ytr.iloc[va], m.decision_function(Ktr[np.ix_(va, tr)])))
        if np.mean(aps) > best_ap:
            best_ap, best_C = float(np.mean(aps)), C
    model = SVC(kernel="precomputed", class_weight="balanced", C=best_C).fit(Ktr, ytr)
    return (model, kernel, sc, Xtr_s), {"C": best_C}


def quantum_scores(bundle, Xte):
    model, kernel, sc, Xtr_s = bundle
    return model.decision_function(kernel.evaluate(sc.transform(Xte), Xtr_s))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def evaluate_all():
    X, y, _, source = load_features()
    Xtr0, Xte0, ytr0, yte0 = split(X, y)
    print(f"data [{source}] | train_cap={TRAIN_CAP} test_cap={TEST_CAP} "
          f"| features={X.shape[1]} | seeds={len(list(SEEDS))} | recall={RECALL:.0%}\n")

    rows = {m: {"pr_auc": [], "fp_elim": [], "clear": [], "clear_err": []}
            for m in ("quantum(angle)", "xgboost(tuned)", "rbf(tuned)")}

    for seed in SEEDS:
        Xtr, ytr = cap_train(Xtr0, ytr0, cap=TRAIN_CAP, seed=seed)
        Xte, yte = cap_train(Xte0, yte0, cap=TEST_CAP, seed=seed)
        qb, qp = tuned_quantum_angle(Xtr, ytr, seed)
        xgb, xp = tuned_xgb(Xtr, ytr, seed)
        rbf, rp = tuned_rbf(Xtr, ytr, seed)
        scored = {
            "quantum(angle)": quantum_scores(qb, Xte),
            "xgboost(tuned)": _ranking_score(xgb, Xte),
            "rbf(tuned)": _ranking_score(rbf, Xte),
        }
        for name, s in scored.items():
            p = _sigmoid((s - s.mean()) / (s.std() + 1e-9))
            rows[name]["pr_auc"].append(average_precision_score(yte, s))
            rows[name]["fp_elim"].append(fp_eliminated_at_recall(yte, s))
            cs = auto_clear_split(yte, p)
            rows[name]["clear"].append(cs["cleared_share"])
            rows[name]["clear_err"].append(cs["cleared_error"])
        print(f"  seed {seed}: params q={qp} xgb={xp} rbf={rp}")

    print(f"\nrandom-guess PR-AUC ~ {float(yte0.mean()):.3f}\n")
    print(f"{'model':<16}{'PR-AUC':>14}{'FP-elim@90%rec':>18}{'auto-clear@<=1%':>18}")
    summary = []
    for name, d in rows.items():
        pr = f"{np.mean(d['pr_auc']):.3f}+/-{np.std(d['pr_auc']):.3f}"
        fp = f"{np.mean(d['fp_elim']):.1%}+/-{np.std(d['fp_elim']):.1%}"
        cl = f"{np.mean(d['clear']):.1%} @ {np.mean(d['clear_err']):.1%}"
        print(f"{name:<16}{pr:>14}{fp:>18}{cl:>18}")
        summary.append({"model": name, "pr_auc": np.mean(d["pr_auc"]),
                        "pr_auc_std": np.std(d["pr_auc"]),
                        "fp_elim_at_90recall": np.mean(d["fp_elim"]),
                        "auto_clear_share": np.mean(d["clear"])})
    pd.DataFrame(summary).to_csv("data/benchmark_results.csv", index=False)
    print("\nwrote data/benchmark_results.csv")


if __name__ == "__main__":
    evaluate_all()
