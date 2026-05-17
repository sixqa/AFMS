"""
Figure: AFMS Stress Test Timeline - Nature-style single panel
Shows full continuous measurement timeline with pause periods.
Legend (right) and stats (top) placed outside to avoid data occlusion.
Output: benchmark/figures/fig_timeline.{svg,pdf,tiff}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import json, os, warnings
warnings.filterwarnings("ignore")

DATA_DIR = r"D:\code\AIFISHMEASURE\AFMS\benchmark\data"
FIG_DIR = r"D:\code\AIFISHMEASURE\AFMS\benchmark\figures"
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "full_dataset.csv"))
with open(os.path.join(DATA_DIR, "pause_points.json")) as f:
    pause_points = json.load(f)

df["sample_idx"] = np.arange(1, len(df)+1)
pause_seqs = {p["sample_seq"] for p in pause_points}
df["is_pause"] = df["sample_idx"].isin(pause_seqs)
df["success_bool"] = (df["measure_success"] == "成功").astype(int)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.linewidth": 0.5, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})

BLUE, GREEN, RED, GREY, LGREY, DARK = "#0F4D92", "#2E8B57", "#D62728", "#8C8C8C", "#E8E8E8", "#333333"
Y_LIMIT = 25

fig, ax = plt.subplots(figsize=(7.8, 2.8))

# 1. Pause shading
for p in pause_points:
    ax.axvspan(p["sample_seq"] - 0.5, p["sample_seq"] + 3.5, color=LGREY, alpha=0.5, linewidth=0)

# 2. Normal samples
m = (df["success_bool"]==1) & (df["process_duration"]<=Y_LIMIT) & (~df["is_pause"])
ax.scatter(df.loc[m,"sample_idx"], df.loc[m,"process_duration"],
           s=0.8, c=BLUE, alpha=0.35, rasterized=True, zorder=2)

# 3. Pause-affected
m = df["is_pause"] & (df["process_duration"]<=Y_LIMIT)
ax.scatter(df.loc[m,"sample_idx"], df.loc[m,"process_duration"],
           s=2.5, c=GREEN, alpha=0.7, zorder=3)

# 4. Failed
m = (df["success_bool"]==0) & (df["process_duration"]<=Y_LIMIT)
ax.scatter(df.loc[m,"sample_idx"], df.loc[m,"process_duration"],
           s=4, marker="x", c=RED, alpha=0.8, linewidths=0.5, zorder=4)

# 5. Rolling median
rm = df["process_duration"].rolling(50, center=True).median()
ax.plot(df["sample_idx"], rm, color=DARK, lw=0.6, alpha=0.8, zorder=5)

# 6. Off-scale markers
for _, o in df[df["process_duration"]>Y_LIMIT].iterrows():
    c = GREEN if o["is_pause"] else "#FF7F0E"
    ax.plot(o["sample_idx"], Y_LIMIT*0.93, marker="v", color=c, markersize=4, lw=0, clip_on=False, zorder=6)

# 7. Stats box → 图外上方 (y>1)
mid = df["process_duration"].median()
q95 = df["process_duration"].quantile(0.95)
sr = df["success_bool"].mean() * 100
ax.text(0.015, 1.06,
        f"n = {len(df)}    Median: {mid:.1f}s    Q95: {q95:.1f}s    Success: {sr:.1f}%",
        transform=ax.transAxes, fontsize=6.5, va="bottom", ha="left",
        color=DARK, zorder=10)

# 8. Legend → 图外右侧
ax.legend(handles=[
    plt.Line2D([0],[0], marker="o", color="w", mfc=BLUE, ms=3, alpha=0.5, label="Normal"),
    plt.Line2D([0],[0], marker="o", color="w", mfc=GREEN, ms=3.5, label="Pause-affected"),
    plt.Line2D([0],[0], marker="x", color=RED, ms=3, lw=0, label="Failed"),
    plt.Line2D([0],[0], color=DARK, lw=0.6, label="Rolling median (w=50)"),
    mpatches.Patch(facecolor=LGREY, alpha=0.5, label="Pause period"),
], loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=6)

# 9. Axis
ax.set_xlabel("Sample sequence", fontsize=7, labelpad=2)
ax.set_ylabel("Processing time (s)", fontsize=7, labelpad=2)
ax.set_xlim(0, len(df)+20)
ax.set_ylim(0, Y_LIMIT)
ax.set_yticks([0, 5, 10, 15, 20, 25])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(pad=1.5)

# 10. Off-scale label
ax.annotate("Off-scale outlier ▲ (up to 450s)", xy=(1,1), xytext=(1,1.035),
            fontsize=5.5, color=GREY, ha="right", va="bottom",
            xycoords="axes fraction", fontstyle="italic", annotation_clip=False)

plt.tight_layout(pad=0.5)
base = os.path.join(FIG_DIR, "fig_timeline")
fig.savefig(f"{base}.svg", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{base}.tiff", dpi=600, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Done: {base}")
