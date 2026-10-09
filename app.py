"""Q-SIEVE analyst dashboard.

For one alert it shows: score & rank, feature attribution, nearest confirmed
cases (precedent), and feature values. Up top: the auto-clear business-case
chart, the calibration check, and the queue. Every decision is written to the
audit log. Backend (classical / quantum) is switchable in the sidebar.

Run:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from data import load_features
from pipeline import build_model
from trust import AuditLog, attribution, precedents, reliability_points

st.set_page_config(page_title="Q-SIEVE", layout="wide")


@st.cache_resource(show_spinner="Training pipeline…")
def run_pipeline(backend: str, max_error: float, difficulty: float):
    X, y, meta, source = load_features(difficulty=difficulty)
    r = build_model(X, y, backend=backend, max_error=max_error)
    p = r["model"].predict_proba(r["Xte"])[:, 1]
    queue = pd.DataFrame(
        {
            "window_id": meta.loc[r["Xte"].index, "window_id"].values,
            "kind": meta.loc[r["Xte"].index, "kind"].values,
            "p_suspicious": p,
            "true_label": r["yte"].values,
        },
        index=r["Xte"].index,
    )
    queue["decision"] = np.where(
        queue.p_suspicious <= r["pick"]["threshold"], "auto_cleared", "left_in_queue")
    queue["rank"] = queue.p_suspicious.rank(ascending=False).astype(int)

    kernel_name = "quantum" if backend == "quantum" else "classical_fallback"
    log = AuditLog(model_version=f"{backend}-cal-v1", kernel=kernel_name)
    for idx, qrow in queue.iterrows():
        log.record(window_id=qrow.window_id, p_suspicious=qrow.p_suspicious,
                   threshold=r["pick"]["threshold"], features=r["Xte"].loc[idx])
    return r, queue, source, log


# --- Sidebar -----------------------------------------------------------------
st.sidebar.title("Q-SIEVE")
st.sidebar.caption("AML alert triage — only ever removes work")
backend = st.sidebar.radio("Classifier backend", ["classical", "quantum"],
                           help="Same pipeline; swap the Layer-2 kernel.")
max_error = st.sidebar.slider("Max error among auto-cleared", 0.0, 0.05, 0.02, 0.005)
difficulty = st.sidebar.slider("Synthetic difficulty", 0.0, 1.0, 0.5, 0.1,
                               help="Class overlap + label noise. Ignored once real data is loaded.")

r, queue, source, log = run_pipeline(backend, max_error, difficulty)
st.sidebar.success(f"Data source: {source}")
if source == "synthetic":
    st.sidebar.info("Synthetic data. Drop data/X.csv to run on real IBM AML data.")

# --- Header metrics ----------------------------------------------------------
st.title("Analyst queue")
pick = r["pick"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Alerts in queue", len(queue))
c2.metric("Auto-cleared", f"{(queue.decision=='auto_cleared').sum()} ({pick['cleared_share']:.0%})")
c3.metric("Error among cleared", f"{pick['error_rate']:.1%}", help=f"target <= {max_error:.1%}")
c4.metric("PR-AUC", f"{r['pr_auc']:.3f}", help="Precision-recall AUC. Accuracy is not used.")
c5.metric("Backend", backend)

# --- Business case + calibration ---------------------------------------------
chart_l, chart_r = st.columns(2)
with chart_l:
    st.subheader("Auto-clear coverage vs error")
    st.line_chart(r["curve"].set_index("cleared_share")[["error_rate"]], y="error_rate")
    st.caption(f"Operating point: clear {pick['cleared_share']:.0%} of the queue at "
               f"{pick['error_rate']:.1%} error (p_suspicious <= {pick['threshold']:.3f}).")
with chart_r:
    st.subheader("Calibration")
    rel = reliability_points(r["model"], r["Xte"], r["yte"])
    if len(rel):
        cal = rel.set_index("mean_pred")[["observed"]].rename(columns={"observed": "observed rate"})
        cal["perfect"] = cal.index
        st.line_chart(cal)
    st.caption("Predicted P(suspicious) against the rate actually observed in each bin.")

# --- Queue + alert detail ----------------------------------------------------
left, right = st.columns([1.2, 1])
with left:
    st.subheader("Queue")
    show = queue.sort_values("rank")[["window_id", "kind", "p_suspicious", "decision", "rank"]].reset_index(drop=True)
    st.dataframe(show.style.format({"p_suspicious": "{:.3f}"}), width="stretch", height=360)
    selected = st.selectbox("Inspect an alert", show.window_id.tolist())

with right:
    st.subheader(f"Alert {selected}")
    row = queue[queue.window_id == selected].iloc[0]
    idx = queue[queue.window_id == selected].index[0]
    d1, d2 = st.columns(2)
    d1.metric("P(suspicious)", f"{row.p_suspicious:.3f}")
    d2.metric("Decision", "✅ auto-cleared" if row.decision == "auto_cleared" else "⏸ left in queue")
    st.caption(f"Queue rank {int(row['rank'])} of {len(queue)} · true kind: {row.kind}")

    x_row = r["Xte"].loc[idx]
    with st.spinner("Scoring this alert…"):
        st.markdown("**What drove this decision**")
        attr = attribution(r["model"], x_row, background=r["Xtr"], n_repeats=8)
        st.bar_chart(attr.head(6))

        st.markdown("**Nearest confirmed cases**")
        base = r["base"]
        if hasattr(base, "similarity"):
            sim_row = base.similarity(x_row.to_frame().T)[0]
            top, sims = precedents(sim_row, base.y_train_, k=3)
            if len(top):
                prec = pd.DataFrame({"training_row": [int(i) for i in top],
                                     "similarity": [float(s) for s in sims]})
                st.caption(f"Resembles {len(top)} confirmed suspicious case(s):")
                st.dataframe(prec.style.format({"similarity": "{:.3f}"}), width="stretch")
            else:
                st.caption("No confirmed suspicious cases nearby.")
        else:
            st.caption("Switch backend to **quantum** — precedent retrieval reuses "
                       "the quantum kernel's similarity.")

    with st.expander("Feature values"):
        st.dataframe(x_row.to_frame("value").style.format("{:.3f}"))

# --- Audit log ---------------------------------------------------------------
st.subheader("Audit log")
st.caption("Every queue decision, reproducible later. The system only records a "
           "removal from the queue — it never escalates or closes a case.")
log_view = pd.DataFrame({
    "window_id": [d.window_id for d in log.records],
    "p_suspicious": [d.p_suspicious for d in log.records],
    "decision": [d.decision for d in log.records],
    "threshold": [d.threshold for d in log.records],
    "kernel": [d.kernel for d in log.records],
})
st.dataframe(log_view.style.format({"p_suspicious": "{:.3f}", "threshold": "{:.3f}"}),
             width="stretch", height=240)
st.download_button("Download audit log", data=log.dumps(),
                   file_name="q-sieve-audit.jsonl", mime="application/jsonl")
