"""Slide-ready charts from the real logged results. -> pitch_assets/*.png"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("pitch_assets", exist_ok=True)
TEAL = "#1A9E8F"; GREEN = "#6FA463"; GREY = "#9AA7A0"; AMBER = "#D99A2B"
NAVY = "#12304A"; RED = "#C0564A"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                     "axes.edgecolor": "#C9D2CC", "axes.linewidth": 1})


def save(fig, name):
    fig.savefig(f"pitch_assets/{name}", dpi=160, bbox_inches="tight",
                facecolor="white"); plt.close(fig)
    print("wrote pitch_assets/" + name)


# 1 — the honest benchmark (classical wins, at first)
fig, ax = plt.subplots(figsize=(8, 4.2))
names = ["XGBoost\n(classical)", "Quantum\n(c=1, untuned)", "RBF kernel", "Random floor"]
vals = [0.190, 0.125, 0.097, 0.093]; cols = [GREEN, TEAL, GREY, "#D7DDD8"]
b = ax.bar(names, vals, color=cols, width=0.62)
for r, v in zip(b, vals):
    ax.text(r.get_x()+r.get_width()/2, v+0.004, f"{v:.3f}", ha="center", fontweight="bold")
ax.axhline(0.093, ls="--", color=RED, lw=1); ax.set_ylim(0, 0.23)
ax.set_ylabel("PR-AUC"); ax.set_title("The honest benchmark: classical wins — at first", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "01_benchmark.png")

# 2 — bandwidth: a tie, not a win (conservative 5-seed / 3000-test numbers, with error bars)
c   = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]
p   = [0.173, 0.181, 0.184, 0.159, 0.152, 0.147, 0.115]
err = [0.049, 0.051, 0.042, 0.043, 0.038, 0.025, 0.017]
fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.axhspan(0.137, 0.197, color=GREEN, alpha=0.16)
ax.axhline(0.167, color=GREEN, lw=1.8, label="classical — tuned XGBoost (0.17 ± 0.03)")
ax.errorbar(c, p, yerr=err, fmt="-o", color=TEAL, lw=2.2, ms=6, capsize=4, label="quantum — angle encoding (± std)")
ax.axvline(0.083, ls="--", color=AMBER, lw=1.6)
ax.annotate("geometric rule\nc = 1/√(n·d) = 0.083", xy=(0.083, 0.205), xytext=(0.15, 0.262),
            color=AMBER, fontsize=10.5, fontweight="bold", ha="left", va="top",
            arrowprops=dict(arrowstyle="->", color=AMBER),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=AMBER, lw=1.2))
ax.annotate("untuned, 0.115", xy=(1.0, 0.115), xytext=(0.60, 0.127), color=TEAL, fontsize=10.5,
            arrowprops=dict(arrowstyle="->", color=TEAL))
ax.set_xlabel("bandwidth  c  (encoding-angle scale)"); ax.set_ylabel("PR-AUC")
ax.set_title("Bandwidth pulls quantum LEVEL with classical — a tie, not a win", fontweight="bold")
ax.set_ylim(0.09, 0.28); ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "02_bandwidth.png")

# 3 — tuned kernel on three real QPUs
vendors = ["IQM Garnet", "IBM ibm_fez", "VTT Q50"]
hw = [0.183, 0.395, 0.265]; sim = 0.231
x = np.arange(len(vendors))
fig, ax = plt.subplots(figsize=(8.4, 4.4))
ax.axhline(sim, color=GREY, lw=1.6, ls="--")
ax.text(2.42, sim+0.004, "simulator 0.231", color="#6B756F", fontsize=11, ha="right")
b = ax.bar(x, hw, color=[TEAL, NAVY, "#3BA99C"], width=0.52)
for r, v in zip(b, hw):
    ax.text(r.get_x()+r.get_width()/2, v+0.004, f"{v:.3f}", ha="center", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(vendors); ax.set_ylim(0, 0.48)
ax.set_ylabel("PR-AUC (hardware)")
ax.set_title("Tuned kernel on three QPUs — scatter, not a held result", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "03_hardware_tuned.png")

# 4 — robustness: zero-drift untuned config across vendors
vendors2 = ["IBM marrakesh", "IQM Garnet", "VTT Q50"]
simv = [0.111, 0.111, 0.111]; hwv = [0.111, 0.111, 0.100]; drift = [0.007, 0.010, 0.018]
x = np.arange(len(vendors2)); w = 0.36
fig, ax = plt.subplots(figsize=(8.4, 4.2))
ax.bar(x-w/2, simv, w, label="simulator", color=GREY)
ax.bar(x+w/2, hwv, w, label="real hardware", color=TEAL)
for i, d in enumerate(drift):
    ax.text(i, 0.125, f"drift {d:.3f}", ha="center", fontsize=10, color="#6B756F")
ax.set_xticks(x); ax.set_xticklabels(vendors2); ax.set_ylim(0, 0.15)
ax.set_ylabel("PR-AUC"); ax.legend(frameon=False, loc="upper right")
ax.set_title("Robust config: near-zero degradation on 3 vendors", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "04_hardware_robust.png")
print("done")
