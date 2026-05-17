#!/usr/bin/env python3
"""
plot_benchmark.py — 绘制 5 物种跨物种对比图
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# Data — best validation metrics from training logs
# ============================================================
species_name = {
    'carp': 'Carp',
    'crab': 'Crab',
    'crayfish': 'Crayfish',
    'ms': 'Mullet',
    'wuli': 'Snakehead'
}
species_label = ['Carp', 'Crab', 'Crayfish', 'Mullet', 'Snakehead']
species_en = ['carp', 'crab', 'crayfish', 'ms', 'wuli']
keypoints_count = [12, 12, 3, 12, 7]

det_mAP = [0.871, 0.762, 0.682, 0.687, 0.787]
det_mAP50 = [0.988, 0.981, 0.955, 0.970, 1.000]

kp_PCK = [0.928, 0.974, 0.687, 0.932, 0.967]
kp_AUC = [0.1228, 0.15, 0.05, 0.13, 0.10]  # approximate from training
# Actually we can extract these from the saved batch_results

# Try to load exact values
batch_json = sorted(ROOT.glob("evals/batch_results_*.json"))
if batch_json:
    data = json.load(batch_json[-1].open())
    det_mAP = [data[f'{s}_det']['bbox_mAP'] for s in species_en]
    det_mAP50 = [data[f'{s}_det']['bbox_mAP_50'] for s in species_en]
    kp_PCK = [data[f'{s}_kp']['PCK'] for s in species_en]

# ============================================================
# Figure 1: Detection mAP + mAP50 bar chart
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(species_label))
width = 0.35

# Detection
ax = axes[0]
bars1 = ax.bar(x - width/2, det_mAP, width, label='mAP', color='#4472C4', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, det_mAP50, width, label='mAP@0.5', color='#ED7D31', edgecolor='white', linewidth=0.5)
for bar, val in zip(bars1, det_mAP):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, det_mAP50):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
            ha='center', va='bottom', fontsize=9)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Detection Performance (RTMDet-Tiny)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(species_label, fontsize=11)
ax.set_ylim(0, 1.12)
ax.legend(fontsize=10, loc='lower right')
ax.grid(axis='y', alpha=0.3)

# Keypoint
ax = axes[1]
colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5']
bars = ax.bar(species_label, kp_PCK, color=colors, edgecolor='white', linewidth=0.5)
for bar, val, kp_n in zip(bars, kp_PCK, keypoints_count):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}\n({kp_n} kpts)', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('PCK', fontsize=12)
ax.set_title('Keypoint Performance (RTMPose-S)', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.12)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
out_path = ROOT / "evals" / "benchmark_comparison.png"
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"[SAVED] {out_path}")
plt.close(fig)

# ============================================================
# Figure 2: Combined radar/scatter — species vs metrics
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(species_label))
ms = 12
ax.scatter(x_pos, det_mAP, s=ms*20, c='#4472C4', marker='s', label='Detection mAP', zorder=3, edgecolors='white', linewidth=0.8)
ax.scatter(x_pos, kp_PCK, s=ms*20, c='#ED7D31', marker='o', label='Keypoint PCK', zorder=3, edgecolors='white', linewidth=0.8)
for i, (dm, km) in enumerate(zip(det_mAP, kp_PCK)):
    ax.annotate(f'{dm:.3f}', (i, dm), textcoords="offset points", xytext=(18, 0),
                fontsize=8, color='#4472C4', fontweight='bold')
    ax.annotate(f'{km:.3f}', (i, km), textcoords="offset points", xytext=(18, 0),
                fontsize=8, color='#ED7D31', fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels([f'{s}\n({k} kpts)' for s, k in zip(species_label, keypoints_count)], fontsize=11)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Cross-Species Detection & Keypoint Performance', fontsize=13, fontweight='bold')
ax.set_ylim(0.5, 1.05)
ax.legend(fontsize=10, loc='lower left')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
out_path = ROOT / "evals" / "cross_species_scatter.png"
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"[SAVED] {out_path}")
plt.close(fig)

# ============================================================
# Figure 3: mAP50 vs PCK with keypoint count annotation
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))
for i, (s, k) in enumerate(zip(species_label, keypoints_count)):
    ax.scatter(det_mAP50[i], kp_PCK[i], s=(k+3)*30, alpha=0.8, label=f'{s} ({k} kpts)',
               edgecolors='black', linewidth=1)
    ax.annotate(s, (det_mAP50[i], kp_PCK[i]), textcoords="offset points",
                xytext=(5, 5), fontsize=9, fontweight='bold')
ax.set_xlabel('Detection mAP@0.5', fontsize=12)
ax.set_ylabel('Keypoint PCK', fontsize=12)
ax.set_title('Detection vs Keypoint Trade-off', fontsize=13, fontweight='bold')
ax.set_xlim(0.94, 1.01)
ax.set_ylim(0.65, 1.0)
ax.legend(fontsize=8, loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
out_path = ROOT / "evals" / "det_vs_kp_tradeoff.png"
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"[SAVED] {out_path}")
plt.close(fig)

print("\n[DONE] All plots saved.")
