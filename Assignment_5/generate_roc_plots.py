"""
Generate ROC plots for Assignment 5 (E1 244 Detection & Estimation Theory).
Run:  python generate_roc_plots.py
Outputs: roc_p2.pdf, roc_p4.pdf, roc_p5.pdf  (in the same directory)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
})

# ── Problem 2 ──────────────────────────────────────────────────────────────
# P_D = P_F (1 - ln P_F)
fig, ax = plt.subplots(figsize=(5, 4))
pf = np.linspace(1e-6, 1, 1000)
pd = pf * (1 - np.log(pf))
pd = np.clip(pd, 0, 1)

ax.plot(pf, pd, color="steelblue", lw=2)
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance line")
ax.set_xlabel(r"$P_F$")
ax.set_ylabel(r"$P_D$")
ax.set_title(r"Problem 2 ROC: $P_D = P_F(1 - \ln P_F)$")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend()
fig.tight_layout()
fig.savefig("roc_p2.pdf")
print("Saved roc_p2.pdf")
plt.close()

# ── Problem 4 ──────────────────────────────────────────────────────────────
# P_D = (-2 + 3*sqrt(1 + 4*P_F)) / 4,   P_F in [0, 0.75]
fig, ax = plt.subplots(figsize=(5, 4))
pf = np.linspace(0, 0.75, 500)
pd = (-2 + 3 * np.sqrt(1 + 4 * pf)) / 4

ax.plot(pf, pd, color="darkorange", lw=2)
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance line")

# Mark the two boundary points
ax.plot(0, 0.25, "o", color="darkorange", ms=6, zorder=5)
ax.plot(0.75, 1, "o", color="darkorange", ms=6, zorder=5)
ax.annotate(r"$(0,\;0.25)$", xy=(0, 0.25), xytext=(0.05, 0.2),
            arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=10)
ax.annotate(r"$(0.75,\;1)$", xy=(0.75, 1), xytext=(0.55, 0.88),
            arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=10)

ax.set_xlabel(r"$P_F$")
ax.set_ylabel(r"$P_D$")
ax.set_title(r"Problem 4 ROC: $P_D = \frac{-2 + 3\sqrt{1+4P_F}}{4}$")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend()
fig.tight_layout()
fig.savefig("roc_p4.pdf")
print("Saved roc_p4.pdf")
plt.close()

# ── Problem 5 ──────────────────────────────────────────────────────────────
# For each lambda: P_D = P_F + lambda/2,  P_F in [0, 1 - lambda/2]
fig, ax = plt.subplots(figsize=(5.5, 4.5))

lambdas   = [0.5, 1.0, 1.5, 2.0]
colors    = ["tab:red", "tab:purple", "tab:green", "tab:blue"]
labels    = [r"$\lambda = 0.5$", r"$\lambda = 1.0$",
             r"$\lambda = 1.5$", r"$\lambda = 2.0$"]

for lam, col, lab in zip(lambdas, colors, labels):
    pf_max = 1 - lam / 2
    pf = np.linspace(0, pf_max, 300)
    pd = pf + lam / 2
    ax.plot(pf, pd, color=col, lw=2, label=lab)
    # Mark endpoints
    ax.plot(0,      lam/2,  "o", color=col, ms=6, zorder=5)
    ax.plot(pf_max, 1,      "o", color=col, ms=6, zorder=5)

ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance line")
ax.set_xlabel(r"$P_F$")
ax.set_ylabel(r"$P_D$")
ax.set_title(r"Problem 5 ROC: $P_D = P_F + \lambda/2$")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("roc_p5.pdf")
print("Saved roc_p5.pdf")
plt.close()

print("All plots generated successfully.")
