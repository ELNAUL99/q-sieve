"""metrics.py — evaluation metrics, one home.

PR-AUC (average_precision_score) is used inline next to the models; this module
holds the AML-specific operative metrics and the noise-aware ranking shared by
the sweeps.
"""

from __future__ import annotations

import numpy as np

RECALL = 0.90        # fixed-recall operating point
SAFE_ERR = 0.01      # max error allowed among auto-cleared


def fp_eliminated_at_recall(y, score, target: float = RECALL) -> float:
    """Share of legitimate alerts scored below the threshold that keeps `target`
    recall on the suspicious class — i.e. false positives the analyst is spared."""
    y = np.asarray(y); score = np.asarray(score)
    pos = score[y == 1]
    if len(pos) == 0:
        return 0.0
    thr = np.quantile(pos, 1 - target)
    neg = score[y == 0]
    return float(np.mean(neg < thr)) if len(neg) else 0.0


def auto_clear_split(y, p, safe_err: float = SAFE_ERR) -> dict:
    """Largest clean share clearable at <= safe_err, and what stays with the
    analyst (the uncertain/retained remainder)."""
    y = np.asarray(y); p = np.asarray(p)
    best = (0.0, 0.0)
    for t in np.linspace(0, 1, 101):
        cleared = p <= t
        if cleared.any():
            err = float(y[cleared].mean())
            if err <= safe_err:
                best = max(best, (float(cleared.mean()), err))
    return {"cleared_share": best[0], "cleared_error": best[1],
            "retained_share": 1 - best[0]}


def rank_within_noise(df, name_col: str, mean_col: str = "mean", std_col: str = "std"):
    """Sort configs by mean descending and tag each: best / ~tie (within noise) /
    worse. A config only beats the best if its mean gap exceeds the combined std."""
    df = df.sort_values(mean_col, ascending=False).reset_index(drop=True)
    best = df.iloc[0]
    df["vs_best"] = df.apply(
        lambda r: "best" if r[name_col] == best[name_col]
        else ("~tie (within noise)"
              if (best[mean_col] - r[mean_col]) <= (best[std_col] + r[std_col])
              else "worse"),
        axis=1,
    )
    return df
