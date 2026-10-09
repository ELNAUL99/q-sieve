"""Data loader — the single seam between synthetic and real data.

The dashboard and experiments call load_features(); they never care whether the
data is synthetic or the real IBM AML dataset. Drop a consolidated feature table
at data/X.csv (FEATURE_COLUMNS + a `label` column) and this returns it.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from consolidate import FEATURE_COLUMNS

DATA_DIR = "data"
FEATURES_CSV = os.path.join(DATA_DIR, "X.csv")


def cap_train(X, y, cap: int, seed: int = 0):
    """Stratified subsample of a training set to `cap` rows (keeps the class
    ratio). Holds every model to the same small, sim-scale training size (the
    quantum kernel's O(n^2) limit)."""
    if len(X) <= cap:
        return X, y
    rng = np.random.default_rng(seed)
    pos = np.where(y.values == 1)[0]
    neg = np.where(y.values == 0)[0]
    n_pos = max(1, round(cap * len(pos) / len(y)))
    n_neg = cap - n_pos
    take = np.concatenate([
        rng.choice(pos, size=min(n_pos, len(pos)), replace=False),
        rng.choice(neg, size=min(n_neg, len(neg)), replace=False),
    ])
    take.sort()
    return X.iloc[take], y.iloc[take]


def load_features(difficulty: float = 0.5, n_each: int = 120):
    """Return (X, y, meta, source). source = "real" if data/X.csv exists, else "synthetic"."""
    if os.path.exists(FEATURES_CSV):
        df = pd.read_csv(FEATURES_CSV)
        missing = [c for c in FEATURE_COLUMNS + ["label"] if c not in df.columns]
        if missing:
            raise ValueError(f"{FEATURES_CSV} missing columns: {missing}")
        X = df[FEATURE_COLUMNS].copy()
        y = df["label"].astype(int)
        if "window_id" in df.columns:
            meta = df[["window_id"]].copy()
        else:
            meta = pd.DataFrame({"window_id": [f"W{i:04d}" for i in range(len(X))]})
        meta["kind"] = df["kind"] if "kind" in df.columns else "unknown"
        return X, y, meta, "real"

    from demo_data import make_demo_queue
    X, y, meta, _ = make_demo_queue(n_each=n_each, difficulty=difficulty)
    return X, y, meta, "synthetic"


if __name__ == "__main__":
    X, y, meta, source = load_features()
    print(f"source={source} | X={X.shape} | positives={int(y.sum())}/{len(y)}")
    print("columns:", list(X.columns))
