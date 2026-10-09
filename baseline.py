"""Classical baseline.

XGBoost + RBF-kernel SVC on the consolidated features. This is the number the
quantum kernel is measured against, so it must be tuned honestly. Both models are
weighted for class imbalance; the metric is PR-AUC (average precision).

Public entry points:
    split(X, y)                 -> stratified 70/30 train/test (the frozen contract)
    train_baselines(Xtr, ytr)   -> {"xgb": model, "rbf": model}
    evaluate(models, Xte, yte)  -> {"xgb": pr_auc, "rbf": pr_auc}
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

RANDOM_STATE = 0
TEST_SIZE = 0.3


def split(X: pd.DataFrame, y: pd.Series):
    """Stratified 70/30 split — the train/test contract from README.md."""
    return train_test_split(
        X, y, stratify=y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )


def train_baselines(Xtr, ytr) -> dict:
    """Fit both classical baselines, each weighted for class imbalance."""
    n_neg = int((ytr == 0).sum())
    n_pos = int((ytr == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
    ).fit(Xtr, ytr)

    # No probability=True: for PR-AUC we rank by decision_function (margin).
    rbf = SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ).fit(Xtr, ytr)

    return {"xgb": xgb, "rbf": rbf}


def _ranking_score(model, Xte):
    """A higher-is-more-positive score for each row, however the model exposes it."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xte)[:, 1]
    return model.decision_function(Xte)


def evaluate(models: dict, Xte, yte) -> dict:
    """PR-AUC (average precision) on the held-out set for each model."""
    return {
        name: float(average_precision_score(yte, _ranking_score(model, Xte)))
        for name, model in models.items()
    }


if __name__ == "__main__":
    import numpy as np
    from consolidate import build_feature_table, _fake_window

    rng = np.random.default_rng(0)
    windows = []
    for _ in range(80):
        windows.append((_fake_window("smurfing", rng), 1))
        windows.append((_fake_window("near_threshold", rng), 1))
        windows.append((_fake_window("legit_busy", rng), 0))
        windows.append((_fake_window("legit_busy", rng), 0))

    X, y = build_feature_table(windows)
    Xtr, Xte, ytr, yte = split(X, y)
    scores = evaluate(train_baselines(Xtr, ytr), Xte, yte)
    print(f"train/test: {len(Xtr)}/{len(Xte)} | positives: {int(y.sum())}/{len(y)}")
    for name, ap in scores.items():
        print(f"{name:>4}  PR-AUC {ap:.3f}")
