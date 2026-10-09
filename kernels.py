"""kernels.py — every quantum kernel / encoding variant, one source of truth.

Separated by TYPE so experiments never re-implement kernels:
  SHARED      to_angles(), amplitude_kernel(), svc_scores()
  FIXED       feature_map(), fixed_kernel()                 (nothing learned)
  TRAINED     trainable_zz_circuit(), train_zz_scales(),
              train_zz_featuremap()                         (SPSA on SVC loss)
  AMPLITUDE   amplitude_kernel(), train_amplitude_scales(),
              pca_reduce()
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import zz_feature_map, z_feature_map
from qiskit_machine_learning.kernels import (
    FidelityStatevectorKernel, TrainableFidelityStatevectorKernel)
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.optimizers import SPSA
from qiskit_machine_learning.utils.loss_functions import SVCLoss

CACHE = 8000


# ===========================================================================
# SHARED
# ===========================================================================
def to_angles(Xtr, Xte, hi=np.pi):
    """Scale features into [0, hi] angles, fitting on train only."""
    sc = MinMaxScaler((0, hi)).fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


def svc_scores(Ktr, ytr, Kte, C=1.0, class_weight="balanced"):
    """Fit a precomputed-kernel SVC and return test decision scores (for PR-AUC)."""
    m = SVC(kernel="precomputed", class_weight=class_weight, C=C).fit(Ktr, ytr)
    return m.decision_function(Kte)


def amplitude_kernel(A, B):
    """Amplitude-encoding fidelity kernel = squared overlap of L2-normalized,
    power-of-2-padded vectors (exact; the sim-only comparison / swap test on HW)."""
    def prep(M):
        n = 1 << int(np.ceil(np.log2(M.shape[1])))
        P = np.zeros((len(M), n))
        P[:, :M.shape[1]] = M
        return P / np.clip(np.linalg.norm(P, axis=1, keepdims=True), 1e-12, None)
    return (prep(A) @ prep(B).T) ** 2


# ===========================================================================
# FIXED  — nothing learned; the angles come only from the data
# ===========================================================================
def feature_map(encoding, d, reps=1, entanglement="linear"):
    """angle -> z_feature_map (no entanglement); zz -> zz_feature_map."""
    if encoding == "angle":
        return z_feature_map(feature_dimension=d, reps=reps)
    if encoding == "zz":
        return zz_feature_map(feature_dimension=d, reps=reps, entanglement=entanglement)
    raise ValueError(f"unknown encoding: {encoding!r}")


def fixed_kernel(encoding, d, reps=1, entanglement="linear"):
    return FidelityStatevectorKernel(
        feature_map=feature_map(encoding, d, reps, entanglement),
        cache_size=CACHE, auto_clear_cache=False)


# ===========================================================================
# TRAINED  — SPSA optimizes circuit parameters against the SVC loss
# ===========================================================================
def trainable_zz_circuit(d, entanglement="linear"):
    """ZZ circuit with a scale per feature + one on the entanglers. At scale=2
    this IS zz_feature_map(reps=1), so training starts from the fixed map."""
    x = ParameterVector("x", d)
    theta = ParameterVector("t", d + 1)
    qc = QuantumCircuit(d)
    for i in range(d):
        qc.h(i)
        qc.p(theta[i] * x[i], i)
    pairs = ([(i, j) for i in range(d) for j in range(i + 1, d)]
             if entanglement == "full" else [(i, i + 1) for i in range(d - 1)])
    for a, b in pairs:
        qc.cx(a, b)
        qc.p(theta[d] * (np.pi - x[a]) * (np.pi - x[b]), b)
        qc.cx(a, b)
    return qc, theta


def train_zz_scales(Xtr_s, ytr, entanglement="linear", maxiter=40):
    """Train the fixed ZZ map's scales (balanced SVCLoss, blocking SPSA from the
    fixed map). Returns a trained kernel."""
    fm, theta = trainable_zz_circuit(Xtr_s.shape[1], entanglement)
    qk = TrainableFidelityStatevectorKernel(
        feature_map=fm, training_parameters=list(theta),
        cache_size=CACHE, auto_clear_cache=False)
    res = QuantumKernelTrainer(
        quantum_kernel=qk, loss=SVCLoss(class_weight="balanced"),
        optimizer=SPSA(maxiter=maxiter, blocking=True),
        initial_point=np.full(len(theta), 2.0)).fit(Xtr_s, ytr)
    return res.quantum_kernel


def train_zz_featuremap(Xtr_s, ytr, reps=1, entanglement="linear", maxiter=25):
    """Train a prepended RY layer in front of a fixed ZZ map. Returns a trained kernel."""
    d = Xtr_s.shape[1]
    tp = ParameterVector("t", d)
    fm0 = QuantumCircuit(d)
    for i in range(d):
        fm0.ry(tp[i], i)
    fm = fm0.compose(zz_feature_map(d, reps=reps, entanglement=entanglement))
    qk = TrainableFidelityStatevectorKernel(feature_map=fm, training_parameters=tp)
    res = QuantumKernelTrainer(
        quantum_kernel=qk, loss="svc_loss",
        optimizer=SPSA(maxiter=maxiter), initial_point=[0.1] * d).fit(Xtr_s, ytr)
    return res.quantum_kernel


# ===========================================================================
# AMPLITUDE  — classical squared-cosine overlap (+ trained / PCA variants)
# ===========================================================================
def _amp_dual_loss(K, y):
    """Same quantity as Qiskit's SVCLoss, with the balanced SVC used for scoring."""
    y = np.asarray(y)
    svc = SVC(kernel="precomputed", class_weight="balanced").fit(K, y)
    dual = np.asarray(svc.dual_coef_[0], dtype=float)
    ksv = K[np.ix_(svc.support_, svc.support_)]
    return float(np.sum(np.abs(dual)) - 0.5 * (dual @ ksv @ dual))


def train_amplitude_scales(Xtr01, ytr, maxiter=40):
    """Train one scale per feature in front of the amplitude overlap (scale=1 is
    the fixed kernel). Returns the learned scale vector theta."""
    x0 = np.ones(Xtr01.shape[1])

    def loss(theta):
        return _amp_dual_loss(amplitude_kernel(Xtr01 * theta, Xtr01 * theta), ytr)

    return SPSA(maxiter=maxiter, blocking=True).minimize(loss, x0).x


def pca_reduce(Xtr, Xte, k):
    """Standardize then PCA to k dims (Meeri's dimensionality reduction)."""
    sc = StandardScaler().fit(Xtr)
    pca = PCA(n_components=k, random_state=0).fit(sc.transform(Xtr))
    return pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))
