"""Synthetic demo data for the dashboard and experiments.

`difficulty` adds class overlap + label noise so models do NOT score a perfect
1.0, which makes the headline charts informative and closer to real AML data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from consolidate import build_feature_table, _fake_window

KINDS = ["smurfing", "near_threshold", "legit_busy"]


def _blend_window(kind_a, kind_b, rng, mix):
    """A window whose transactions are drawn partly from another behaviour —
    the ambiguity a real tier-1 analyst actually struggles with."""
    a = _fake_window(kind_a, rng)
    b = _fake_window(kind_b, rng)
    n_b = int(len(a) * mix)
    if n_b and len(b) >= n_b:
        a = pd.concat([a.iloc[: len(a) - n_b], b.iloc[:n_b]], ignore_index=True)
    return a


def make_demo_queue(n_each: int = 120, seed: int = 1, difficulty: float = 0.0):
    """Return (X, y, meta, txns_list).

    n_each      : iterations; each adds 2 positives + 3 negatives.
    difficulty  : 0.0 = cleanly separable; higher = more overlap + label noise.
    """
    rng = np.random.default_rng(seed)
    plan = (
        [("smurfing", 1)]
        + [("near_threshold", 1)]
        + [("legit_busy", 0)] * 3  # realistic imbalance: mostly legitimate
    )
    windows, kinds, txns_list = [], [], []
    for _ in range(n_each):
        for kind, label in plan:
            if difficulty > 0 and rng.random() < difficulty:
                other = "legit_busy" if label == 1 else rng.choice(["smurfing", "near_threshold"])
                txns = _blend_window(kind, other, rng, mix=0.35 + 0.3 * difficulty)
            else:
                txns = _fake_window(kind, rng)
            windows.append((txns, label))
            kinds.append(kind)
            txns_list.append(txns)

    X, y = build_feature_table(windows)

    if difficulty > 0:  # label noise: adjudication is imperfect in real AML
        flip = rng.random(len(y)) < (0.08 * difficulty)
        y = y.copy()
        y[flip] = 1 - y[flip]

    meta = pd.DataFrame(
        {"window_id": [f"W{i:04d}" for i in range(len(X))], "kind": kinds}
    )
    return X, y, meta, txns_list
