"""Layer 3 — Trust layer (the real deliverable).

Wraps a trained classifier so the system only ever *removes* work. It auto-clears
the confidently-legitimate; everything else stays in the queue, untouched. It
never escalates and never closes a suspicious case, so the only risk is a wrong
auto-clear — the headline metric is the error rate among cleared items.

Parts: calibration, abstain/auto-clear curve, audit log, precedent, attribution.
Convention: label 1 = suspicious, 0 = legitimate. Auto-clear when p_suspicious is
very low.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score


# ---------------------------------------------------------------------------
# 1. Calibration
# ---------------------------------------------------------------------------
def calibrate(model, Xcal, ycal, method: str = "isotonic"):
    """Return a calibrated wrapper of an already-fitted model (predict_proba
    becomes calibrated P(suspicious))."""
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    calibrated.fit(Xcal, ycal)
    return calibrated


def reliability_points(model, Xte, yte, n_bins: int = 10):
    """(mean predicted p, observed fraction) per confidence bin, for a reliability plot."""
    p = model.predict_proba(Xte)[:, 1]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            rows.append((float(p[mask].mean()), float(np.asarray(yte)[mask].mean()), int(mask.sum())))
    return pd.DataFrame(rows, columns=["mean_pred", "observed", "count"])


# ---------------------------------------------------------------------------
# 2. Abstain band + auto-clear curve
# ---------------------------------------------------------------------------
def auto_clear_curve(model, Xte, yte, n_points: int = 50) -> pd.DataFrame:
    """Sweep the auto-clear threshold and report coverage vs error among cleared."""
    p = model.predict_proba(Xte)[:, 1]
    y = np.asarray(yte)
    thresholds = np.linspace(0.0, 1.0, n_points + 1)
    rows = []
    for t in thresholds:
        cleared = p <= t
        n_cleared = int(cleared.sum())
        share = n_cleared / len(y)
        err = float(y[cleared].mean()) if n_cleared else 0.0
        rows.append((float(t), share, err, n_cleared))
    return pd.DataFrame(rows, columns=["threshold", "cleared_share", "error_rate", "n_cleared"])


def choose_threshold(curve: pd.DataFrame, max_error: float = 0.01) -> dict:
    """Most-coverage threshold whose error among cleared stays <= max_error."""
    ok = curve[curve.error_rate <= max_error]
    if len(ok):
        best = ok.sort_values("cleared_share").iloc[-1]   # most coverage within budget
    else:
        best = curve.loc[curve.error_rate.idxmin()]       # budget infeasible: lowest error
    return {
        "threshold": float(best.threshold),
        "cleared_share": float(best.cleared_share),
        "error_rate": float(best.error_rate),
        "max_error": max_error,
    }


# ---------------------------------------------------------------------------
# 3. Audit log
# ---------------------------------------------------------------------------
@dataclass
class Decision:
    window_id: str
    p_suspicious: float
    decision: str
    threshold: float
    model_version: str
    kernel: str
    features: dict
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class AuditLog:
    model_version: str
    kernel: str
    records: list = field(default_factory=list)

    def record(self, window_id, p_suspicious, threshold, features) -> Decision:
        decision = "auto_cleared" if p_suspicious <= threshold else "left_in_queue"
        d = Decision(
            window_id=str(window_id),
            p_suspicious=float(p_suspicious),
            decision=decision,
            threshold=float(threshold),
            model_version=self.model_version,
            kernel=self.kernel,
            features={k: float(v) for k, v in dict(features).items()},
        )
        self.records.append(d)
        return d

    def dumps(self) -> str:
        """JSONL text, one decision per line — the same bytes to_jsonl writes."""
        return "".join(json.dumps(asdict(d)) + "\n" for d in self.records)

    def to_jsonl(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.dumps())


# ---------------------------------------------------------------------------
# 5. Precedent retrieval
# ---------------------------------------------------------------------------
def precedents(similarity_row, y_train, k: int = 3, only_confirmed: bool = True):
    """Most similar training windows to one query window (reuses the kernel's
    similarity). Returns (indices, similarities), most similar first."""
    sim = np.asarray(similarity_row, dtype=float)
    y = np.asarray(y_train)
    order = np.argsort(sim)[::-1]
    if only_confirmed:
        order = order[y[order] == 1]
    top = order[:k]
    return top, sim[top]


# ---------------------------------------------------------------------------
# 4. Feature attribution (model-agnostic permutation)
# ---------------------------------------------------------------------------
def attribution(model, x_row: pd.Series, background: pd.DataFrame, n_repeats: int = 20, seed: int = 0) -> pd.Series:
    """How much each feature moves this row's P(suspicious) when shuffled out.
    Works unchanged on a quantum kernel: it only touches inputs and the score."""
    rng = np.random.default_rng(seed)
    base = float(model.predict_proba(x_row.to_frame().T)[:, 1][0])
    impact = {}
    for col in x_row.index:
        deltas = []
        for _ in range(n_repeats):
            perturbed = x_row.copy()
            perturbed[col] = background[col].sample(1, random_state=int(rng.integers(1e9))).iloc[0]
            p = float(model.predict_proba(perturbed.to_frame().T)[:, 1][0])
            deltas.append(abs(p - base))
        impact[col] = float(np.mean(deltas))
    return pd.Series(impact).sort_values(ascending=False)


if __name__ == "__main__":
    from consolidate import build_feature_table, _fake_window, FEATURE_COLUMNS
    from baseline import split, train_baselines

    rng = np.random.default_rng(1)
    windows = []
    for _ in range(120):
        windows.append((_fake_window("smurfing", rng), 1))
        windows.append((_fake_window("near_threshold", rng), 1))
        windows.append((_fake_window("legit_busy", rng), 0))
        windows.append((_fake_window("legit_busy", rng), 0))
        windows.append((_fake_window("legit_busy", rng), 0))

    X, y = build_feature_table(windows)
    Xtr, Xte, ytr, yte = split(X, y)
    Xfit, Xcal, yfit, ycal = split(Xtr, ytr)
    raw = train_baselines(Xfit, yfit)["xgb"]
    model = calibrate(raw, Xcal, ycal)

    print("PR-AUC:", round(average_precision_score(yte, model.predict_proba(Xte)[:, 1]), 3))
    curve = auto_clear_curve(model, Xte, yte)
    pick = choose_threshold(curve, max_error=0.01)
    print(f"auto-clear {pick['cleared_share']:.0%} at {pick['error_rate']:.1%} error")
