"""Phase 1 — build the real feature table from the IBM AML dataset.

Turns raw HI-Small_Trans.csv into data/X.csv (FEATURE_COLUMNS + label + window_id).

Pipeline: load tx -> account-centric signed rows -> (account, ISO week) windows
-> velocity-flag (>= MIN_TX) -> label (any laundering tx = 1) -> keep all
suspicious + capped legitimate -> consolidate() each. Amounts are converted to
USD first (the data mixes 15 currencies).

Run:  python build_features.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from consolidate import consolidate, BASE_FEATURES, FEATURE_COLUMNS

RAW = os.path.join("data", "HI-Small_Trans.csv")
OUT = os.path.join("data", "X.csv")

MIN_TX = 8        # velocity threshold: a window must have >= this many tx to be "flagged"
NEG_CAP = 40000   # cap legitimate flagged windows kept (~9% prevalence; positives all kept)
SEED = 0

# Approximate 2022 USD exchange rates, so amounts are comparable across currencies.
FX_TO_USD = {
    "US Dollar": 1.00, "Euro": 1.05, "Swiss Franc": 1.05, "UK Pound": 1.24,
    "Canadian Dollar": 0.77, "Australian Dollar": 0.69, "Yuan": 0.149,
    "Yen": 0.0076, "Rupee": 0.0127, "Ruble": 0.0145, "Shekel": 0.30,
    "Mexican Peso": 0.050, "Saudi Riyal": 0.27, "Brazil Real": 0.19,
    "Bitcoin": 20000.0,
}


def load_account_centric(path: str) -> pd.DataFrame:
    cols = ["Timestamp", "Account", "Account.1", "Amount Paid", "Payment Currency",
            "Amount Received", "Receiving Currency", "Is Laundering"]
    df = pd.read_csv(
        path, usecols=cols,
        dtype={"Account": "string", "Account.1": "string",
               "Amount Paid": "float64", "Amount Received": "float64",
               "Payment Currency": "category", "Receiving Currency": "category",
               "Is Laundering": "int8"},
    )
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    paid_usd = df["Amount Paid"] * df["Payment Currency"].map(FX_TO_USD).astype("float64")
    recv_usd = df["Amount Received"] * df["Receiving Currency"].map(FX_TO_USD).astype("float64")
    print(f"  loaded {len(df):,} transactions "
          f"({df['Is Laundering'].sum():,} laundering); amounts -> USD")

    deb = pd.DataFrame({
        "account": df["Account"], "amount": -paid_usd, "timestamp": df["Timestamp"],
        "counterparty": df["Account.1"], "is_laundering": df["Is Laundering"]})
    cred = pd.DataFrame({
        "account": df["Account.1"], "amount": recv_usd, "timestamp": df["Timestamp"],
        "counterparty": df["Account"], "is_laundering": df["Is Laundering"]})
    long = pd.concat([deb, cred], ignore_index=True)
    iso = long["timestamp"].dt.isocalendar()
    long["window_id"] = (
        long["account"].astype(str) + "|"
        + iso["year"].astype(str) + "W" + iso["week"].astype(str).str.zfill(2))
    print(f"  exploded to {len(long):,} account-centric rows")
    return long


def select_flagged_windows(long: pd.DataFrame):
    g = long.groupby("window_id")
    size = g.size()
    is_pos = g["is_laundering"].max() == 1
    flagged = size[size >= MIN_TX].index
    pos = flagged.intersection(is_pos[is_pos].index)
    neg = flagged.difference(is_pos[is_pos].index)
    print(f"  flagged windows (>= {MIN_TX} tx): {len(flagged):,} "
          f"({len(pos):,} suspicious, {len(neg):,} legitimate)")
    return pos, neg


def build():
    rng = np.random.default_rng(SEED)
    print("loading…")
    long = load_account_centric(RAW)
    print("selecting flagged windows…")
    pos, neg = select_flagged_windows(long)
    if len(neg) > NEG_CAP:
        neg = pd.Index(rng.choice(neg.to_numpy(), size=NEG_CAP, replace=False))
    keep = pos.union(neg)
    sub = long[long["window_id"].isin(keep)]

    # Account baseline: typical window size across ALL the account's windows.
    win_size = long.groupby("window_id").size()
    acct_of_win = win_size.index.to_series().str.split("|").str[0]
    acct_baseline = win_size.groupby(acct_of_win.values).median()
    print(f"  consolidating {len(keep):,} windows "
          f"({len(pos):,} positive + {len(neg):,} negative)…")

    rows, labels, wids, vratio = [], [], [], []
    for wid, txns in sub.groupby("window_id"):
        rows.append(consolidate(txns))
        labels.append(int(txns["is_laundering"].max()))
        wids.append(wid)
        base = acct_baseline.get(wid.split("|")[0], len(txns))
        vratio.append(len(txns) / max(float(base), 1.0))

    X = pd.DataFrame(rows, columns=BASE_FEATURES)
    X["velocity_ratio"] = vratio
    X = X[FEATURE_COLUMNS]
    X["label"] = labels
    X["window_id"] = wids
    X = X.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    X.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}: {X.shape[0]:,} windows, {int(X['label'].sum()):,} positive "
          f"({X['label'].mean():.1%} prevalence)")
    print("\nviability — n_tx by class:")
    print(X.groupby("label")["n_tx"].describe()[["mean", "25%", "50%", "75%", "max"]]
          .round(1).to_string())


if __name__ == "__main__":
    build()
