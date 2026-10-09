"""Layer 1 — Consolidation.

Turns one flagged (account, window) slice of transactions into a single row of
behavioural features, per the feature-table contract in README.md.

Pure pandas, no model. Public entry points:
    consolidate(txns)            -> dict of 11 within-window features
    build_feature_table(windows) -> (X: DataFrame, y: Series)

Input transaction schema (one row per transaction):
    amount        float   signed; >0 = credit (money in), <0 = debit (money out)
    timestamp     datetime
    counterparty  hashable (account id of the other party)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The selected feature columns — this list IS the contract (see README.md).
# The number of columns here fixes the qubit count in Layer 2, so keep it stable.
# Per-window features computed by consolidate() (pure within-window):
BASE_FEATURES = [
    "n_tx",
    "frag_ratio",
    "in_out_count",
    "in_out_value",
    "pass_through",
    "fan_in",
    "fan_out",
    "round_share",
    "near_10k",
    "near_1k",
    "iat_entropy",
]

# velocity_ratio is cross-window (window n_tx vs the account's own baseline), so
# it is added during table assembly, not inside consolidate().
FEATURE_COLUMNS = BASE_FEATURES + ["velocity_ratio"]


def _entropy(x: pd.Series, bins: int = 10) -> float:
    """Shannon entropy of a 1-D distribution. Low = regular/automated timing."""
    if len(x) < 2:
        return 0.0
    hist, _ = np.histogram(x, bins=bins)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log(p)).sum())


def consolidate(txns: pd.DataFrame) -> dict:
    """Collapse one (account, window) slice of transactions into one feature row."""
    deb = txns[txns.amount < 0]            # money out
    cred = txns[txns.amount > 0]           # money in
    gross = txns.amount.abs().sum()        # total money moved
    gaps = txns.timestamp.sort_values().diff().dt.total_seconds().dropna()

    return {
        "n_tx": float(len(txns)),
        # low frag_ratio = lots of tiny transactions = smurfing tell
        "frag_ratio": gross / max(len(txns), 1),
        "in_out_count": len(cred) / max(len(deb), 1),
        "in_out_value": cred.amount.sum() / max(deb.amount.abs().sum(), 1.0),
        # pass_through ~0 = money in and straight back out = conduit/mule
        "pass_through": abs(txns.amount.sum()) / max(gross, 1.0),
        "fan_in": float(cred.counterparty.nunique()),
        "fan_out": float(deb.counterparty.nunique()),
        "round_share": float((txns.amount.abs() % 100 == 0).mean()),
        "near_10k": float(txns.amount.abs().between(9000, 10000).mean()),
        "near_1k": float(txns.amount.abs().between(900, 1000).mean()),
        "iat_entropy": _entropy(gaps),
    }


def build_feature_table(windows) -> tuple[pd.DataFrame, pd.Series]:
    """Run consolidate() over many labelled windows.

    windows: iterable of (txns_dataframe, label) pairs, where label is 1
             (suspicious) or 0 (legitimate).
    """
    rows, labels = [], []
    for txns, label in windows:
        rows.append(consolidate(txns))
        labels.append(int(label))
    X = pd.DataFrame(rows, columns=BASE_FEATURES)
    # Synthetic data has no account continuity, so approximate velocity_ratio as
    # this window's activity vs the median window. (Real data uses the account's
    # own baseline — see build_features.py.)
    X["velocity_ratio"] = X["n_tx"] / max(X["n_tx"].median(), 1.0)
    X = X[FEATURE_COLUMNS]
    y = pd.Series(labels, name="label")
    return X, y


# --------------------------------------------------------------------------
# Smoke test on synthetic data. Run: python consolidate.py
# --------------------------------------------------------------------------
def _fake_window(kind: str, rng: np.random.Generator) -> pd.DataFrame:
    """Build a toy transaction window of a given behavioural 'kind'."""
    base = pd.Timestamp("2026-01-01")
    if kind == "smurfing":  # one big sum leaves as many tiny debits
        n = rng.integers(80, 200)
        amount = -rng.uniform(50, 400, n)
        cp = rng.integers(0, 150, n)
        secs = np.cumsum(rng.uniform(30, 120, n))
    elif kind == "near_threshold":  # repeated credits just below 10k
        n = rng.integers(8, 20)
        amount = rng.uniform(9200, 9900, n)
        cp = rng.integers(0, 3, n)
        secs = np.cumsum(rng.uniform(3600, 86400, n))
    else:  # "legit_busy": busy but innocent, mixed flow, accumulates
        n = rng.integers(40, 120)
        amount = rng.normal(0, 2000, n)
        cp = rng.integers(0, 60, n)
        secs = np.cumsum(rng.uniform(60, 7200, n))
    return pd.DataFrame(
        {
            "amount": amount,
            "timestamp": base + pd.to_timedelta(secs, unit="s"),
            "counterparty": cp,
        }
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    windows = []
    for _ in range(40):
        windows.append((_fake_window("smurfing", rng), 1))
        windows.append((_fake_window("near_threshold", rng), 1))
        windows.append((_fake_window("legit_busy", rng), 0))
    X, y = build_feature_table(windows)
    print("X shape:", X.shape, "| positives:", int(y.sum()), "/", len(y))
    print("NaNs:", int(X.isna().sum().sum()))
