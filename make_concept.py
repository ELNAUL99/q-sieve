"""Concept diagram: the kernel = overlap, and bandwidth = the concentration knob."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("pitch_assets", exist_ok=True)
NAVY = "#102B32"; TEAL = "#1A9E8F"; RED = "#C0564A"; GREY = "#5B6B64"; LINE = "#C9D2CC"
plt.rcParams["font.family"] = "DejaVu Sans"

fig = plt.figure(figsize=(10, 5.4))
fig.patch.set_facecolor("white")

# ── the one equation
fig.text(0.5, 0.93, r"$K(x,z)\;=\;|\langle\,\phi(x)\,|\,\phi(z)\,\rangle|^{2}$",
         ha="center", va="top", fontsize=30, color=NAVY)
fig.text(0.5, 0.80, "similarity of two accounts  =  overlap of their quantum states"
         "    ( 1 = identical,   0 = completely different )",
         ha="center", va="top", fontsize=12.5, color=GREY)

# ── two unit circles: small vs large bandwidth
def circle(ax, angles, color, title, caption):
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.7, 1.3)
    th = np.linspace(0, 2 * np.pi, 240)
    ax.plot(np.cos(th), np.sin(th), color=LINE, lw=1.4)
    ax.scatter(np.cos(angles), np.sin(angles), s=140, color=color, zorder=3,
               edgecolor="white", linewidth=1.5)
    ax.set_title(title, fontsize=13.5, color=NAVY, fontweight="bold", pad=6)
    ax.text(0, -1.52, caption, ha="center", va="top", fontsize=11.5, color=color)

axL = fig.add_axes([0.04, 0.04, 0.44, 0.66])
axR = fig.add_axes([0.52, 0.04, 0.44, 0.66])
circle(axL, np.array([0.04, 0.11, 0.18, 0.25, 0.33]), TEAL,
       "small bandwidth  (c ≈ 0.1  —  tuned)",
       "states stay close → the kernel\nsees the real differences   ✓")
circle(axR, np.array([0.3, 1.5, 2.8, 4.0, 5.3]), RED,
       "large bandwidth  (c = 1  —  untuned)",
       "states spread out → all look equally\ndifferent → the kernel goes blind   ✗")

fig.savefig("pitch_assets/05_kernel_concept.png", dpi=160, bbox_inches="tight", facecolor="white")
print("wrote pitch_assets/05_kernel_concept.png")
