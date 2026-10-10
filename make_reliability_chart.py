"""Honest hardware chart: tuned config is noise-dominated, robust config is rock-solid."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("pitch_assets", exist_ok=True)
NAVY = "#102B32"; TEAL = "#1A9E8F"; RED = "#C0564A"; GREY = "#8A978F"; AMBER = "#C9962B"
plt.rcParams["font.family"] = "DejaVu Sans"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 5.0), sharey=True,
                               gridspec_kw={"wspace": 0.35})
fig.patch.set_facecolor("white")

# ── left: tuned c=0.1 — three full-shot runs, scattered
tuned = [("IQM Garnet", 0.183), ("VTT Q50", 0.265), ("IBM ibm_fez", 0.395)]
xs = [1, 2, 3]
axL.axhline(0.231, ls="--", color=GREY, lw=1.6)
axL.text(3.5, 0.231, "noiseless\nsim 0.231", color="#6B756F", fontsize=10, va="center", ha="left")
for (name, v), x in zip(tuned, xs):
    col = RED if v > 0.231 else TEAL
    axL.scatter([x], [v], s=200, color=col, edgecolor="white", linewidth=1.5, zorder=3)
    axL.text(x, v + 0.016, f"{v:.3f}", ha="center", fontsize=12, fontweight="bold", color=NAVY)
    axL.text(x, 0.02, name, ha="center", fontsize=10, color="#6B756F", rotation=0)
axL.annotate("above the noiseless sim →\nnoise, not a win", (3, 0.395), (1.25, 0.375),
             fontsize=10.5, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
axL.set_title("Tuned (c=0.1): noise-dominated\nscatter 0.18–0.40",
              fontsize=12.5, color=NAVY, fontweight="bold")
axL.set_ylabel("PR-AUC on hardware", fontsize=12)
axL.set_xlim(0.3, 4.7); axL.set_xticks([])
axL.spines[["top", "right", "bottom"]].set_visible(False)

# ── right: robust c=1 — tight
robust = [("IQM Garnet", 0.111), ("IBM Marrakesh", 0.111), ("VTT Q50", 0.100)]
xr = [1, 2, 3]
axR.axhline(0.111, ls="--", color=GREY, lw=1.6)
axR.text(3.45, 0.111, "sim 0.111", color="#6B756F", fontsize=10, va="center", ha="left")
for (name, v), x in zip(robust, xr):
    axR.scatter([x], [v], s=190, color=TEAL, edgecolor="white", linewidth=1.5, zorder=3)
    axR.text(x, v + 0.016, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold", color=NAVY)
    axR.text(x, 0.02, name, ha="center", fontsize=9, color="#6B756F")
axR.annotate("near-zero drift\n(0.007–0.018)", (2, 0.111), (1.2, 0.30),
             fontsize=10.5, color=TEAL, arrowprops=dict(arrowstyle="->", color=TEAL))
axR.set_title("Robust (c=1): reliable\nholds within ±0.01",
              fontsize=13, color=NAVY, fontweight="bold")
axR.set_xlim(0.3, 4.6); axR.set_xticks([])
axR.spines[["top", "right", "bottom", "left"]].set_visible(False)

fig.suptitle("Same kernel on real hardware: tuning buys simulator score but costs on-device stability",
             fontsize=13.5, color=NAVY, fontweight="bold", y=1.02)
axL.set_ylim(0, 0.43)
fig.savefig("pitch_assets/06_hardware_reliability.png", dpi=160, bbox_inches="tight", facecolor="white")
print("wrote pitch_assets/06_hardware_reliability.png")
