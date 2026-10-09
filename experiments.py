"""Headline experiments → the two charts the deck is built around.

1. Label-ablation curve — PR-AUC vs training positives, quantum vs classical.
2. Auto-clear curve — queue cleared vs error among cleared (the business case).

Run:  python experiments.py   (writes charts/ablation.png and charts/autoclear.png)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

from baseline import split, train_baselines, evaluate
from trust import auto_clear_curve, choose_threshold
from data import load_features

CHARTS_DIR = "charts"
LABEL_COUNTS = [25, 50, 100, 200]
NEG_TRAIN_CAP = 300


def _subsample_positives(Xtr, ytr, n_pos, rng):
    pos_idx = np.where(ytr.values == 1)[0]
    neg_idx = np.where(ytr.values == 0)[0]
    if len(pos_idx) > n_pos:
        pos_idx = rng.choice(pos_idx, size=n_pos, replace=False)
    if len(neg_idx) > NEG_TRAIN_CAP:
        neg_idx = rng.choice(neg_idx, size=NEG_TRAIN_CAP, replace=False)
    keep = np.sort(np.concatenate([pos_idx, neg_idx]))
    return Xtr.iloc[keep], ytr.iloc[keep]


def label_ablation(X, y, label_counts=LABEL_COUNTS, seed=0) -> pd.DataFrame:
    from quantum_kernel import QuantumKernelClassifier
    rng = np.random.default_rng(seed)
    Xtr, Xte, ytr, yte = split(X, y)
    rows = []
    for n_pos in label_counts:
        if (ytr.values == 1).sum() < n_pos:
            continue
        Xs, ys = _subsample_positives(Xtr, ytr, n_pos, rng)
        qk = QuantumKernelClassifier().fit(Xs, ys)
        q_ap = average_precision_score(yte, qk.decision_function(Xte))
        cls = evaluate(train_baselines(Xs, ys), Xte, yte)
        rows.append({"n_pos": n_pos, "quantum": q_ap, "xgb": cls["xgb"], "rbf": cls["rbf"]})
        print(f"  n_pos={n_pos:>3}: quantum {q_ap:.3f} | xgb {cls['xgb']:.3f} | rbf {cls['rbf']:.3f}")
    return pd.DataFrame(rows)


def plot_ablation(df, path):
    plt.figure(figsize=(7, 4.5))
    for col, label, style in [("quantum", "Quantum kernel", "o-"),
                              ("xgb", "XGBoost", "s--"), ("rbf", "RBF-SVC", "^--")]:
        plt.plot(df.n_pos, df[col], style, label=label, linewidth=2, markersize=7)
    plt.xlabel("Training positives (label count)")
    plt.ylabel("PR-AUC on held-out test")
    plt.title("Label-ablation: does quantum hold up as labels shrink?")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


def plot_autoclear(curve, pick, path):
    plt.figure(figsize=(7, 4.5))
    plt.plot(curve.cleared_share, curve.error_rate, "-", linewidth=2, color="#2563eb")
    plt.scatter([pick["cleared_share"]], [pick["error_rate"]], color="#dc2626", zorder=5,
                label=f"operating point: clear {pick['cleared_share']:.0%} @ {pick['error_rate']:.1%} error")
    plt.axhline(pick["max_error"], color="#dc2626", linestyle=":", alpha=0.6,
                label=f"target max error {pick['max_error']:.0%}")
    plt.xlabel("Share of queue auto-cleared")
    plt.ylabel("Error rate among cleared items")
    plt.title("Auto-clear coverage vs error (the business case)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)
    X, y, _, source = load_features()
    print(f"dataset [{source}]: {len(X)} windows, {int(y.sum())} positives, {X.shape[1]} features")
    print("label-ablation:")
    abl = label_ablation(X, y)
    plot_ablation(abl, os.path.join(CHARTS_DIR, "ablation.png"))
    from pipeline import build_model
    r = build_model(X, y, backend="classical", max_error=0.02)
    plot_autoclear(r["curve"], r["pick"], os.path.join(CHARTS_DIR, "autoclear.png"))
    print(f"\nwrote {CHARTS_DIR}/ablation.png and {CHARTS_DIR}/autoclear.png")


if __name__ == "__main__":
    main()
