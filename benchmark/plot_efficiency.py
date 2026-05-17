"""..."""
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

df["sample_idx"] = np.arange(1, len(df) + 1)
pause_seqs = {p["sample_seq"] for p in pause_points}
df["is_pause"] = df["sample_idx"].isin(pause_seqs)
df["success_bool"] = (df["measure_success"] == "成功").astype(int)

# Cumulative time
df["cum_time_min"] = df["process_duration"].cumsum() / 60.0
total_min = df["cum_time_min"].iloc[-1]

# Rolling throughput (5-min window)
window_min = 5
df["throughput"] = np.nan
for i in range(len(df)):
    t_cur = df["cum_time_min"].iloc[i]
    t_start = t_cur - window_min
    if t_start <= 0:
        continue
    mask = df["cum_time_min"] >= t_start
    idx_start = mask.idxmax() if mask.any() else 0
    cnt = i - idx_start + 1
    actual = df["cum_time_min"].iloc[i] - df["cum_time_min"].iloc[idx_start]
    if actual > 0:
        df.at[df.index[i], "throughput"] = cnt / actual

# Plot
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.linewidth": 0.5, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})

BLUE, ORANGE, GREY, LGREY, DARK = "#0F4D92", "#E07A2F", "#8C8C8C", "#E8E8E8", "#333333"

fig, ax1 = plt.subplots(figsize=(7.2, 2.8))

# Pause bands
for p in pause_points:
    seq = p["sample_seq"]
    if seq in df["sample_idx"].values:
        t_start = df.loc[df["sample_idx"] == seq, "cum_time_min"].values[0]
        end_seq = min(seq + 4, len(df))
        t_end = df.loc[df["sample_idx"] == end_seq, "cum_time_min"].values[0]
        ax1.axvspan(t_start, t_end, color=LGREY, alpha=0.5, linewidth=0, zorder=1)

# Cumulative samples curve
ax1.plot(df["cum_time_min"], df["sample_idx"], color=BLUE, lw=0.8, zorder=3)
ax1.fill_between(df["cum_time_min"], 0, df["sample_idx"], color=BLUE, alpha=0.08, zorder=2)

# Throughput secondary axis
ax2 = ax1.twinx()
ax2.plot(df["cum_time_min"], df["throughput"], color=ORANGE, lw=0.5, alpha=0.7, zorder=4)
smooth = df["throughput"].rolling(30, center=True, min_periods=1).mean()
ax2.plot(df["cum_time_min"], smooth, color=ORANGE, lw=1.0, zorder=5)

# Labels outside
ax1.text(0.02, 1.06, 
         f"Total: {len(df)} samples in {total_min:.0f} min ({total_min/60:.2f}h)",
         transform=ax1.transAxes, fontsize=6.5, va="bottom", ha="left", color=DARK)
ax1.text(0.02, 1.12,
         f"Avg throughput: {len(df)/total_min:.1f} samples/min",
         transform=ax1.transAxes, fontsize=6.5, va="bottom", ha="left", color=DARK)
ax1.text(0.02, 1.18,
         f"Pause periods: {len(pause_points)} times  |  Success: {df['success_bool'].mean()*100:.1f}%",
         transform=ax1.transAxes, fontsize=6.5, va="bottom", ha="left", color=GREY)

# Legend outside
legend_elements = [
    plt.Line2D([0], [0], color=BLUE, lw=0.8, label="Cumulative samples"),
    plt.Line2D([0], [0], color=ORANGE, lw=1.0, label="Throughput (5min rolling)"),
    mpatches.Patch(facecolor=LGREY, alpha=0.5, label="Pause period"),
]
ax1.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1.0),
           frameon=False, fontsize=6)

ax1.set_xlabel("Elapsed time (min)", fontsize=7, labelpad=2)
ax1.set_ylabel("Cumulative samples processed", fontsize=7, labelpad=2)
ax2.set_ylabel("Throughput (samples/min)", fontsize=7, labelpad=2, color=ORANGE)

ax1.set_xlim(0, total_min * 1.02)
ax1.set_ylim(0, len(df) * 1.03)
ax2.set_ylim(0, max(df["throughput"].dropna()) * 1.2)

ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_color(ORANGE)

plt.tight_layout(pad=0.5)
base = os.path.join(FIG_DIR, "fig_efficiency")
fig.savefig(f"{base}.svg", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{base}.tiff", dpi=600, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Done: {base}")
