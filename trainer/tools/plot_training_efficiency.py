#!/usr/bin/env python3
"""
plot_training_efficiency.py — Visualise rapid adaptation: convergence speed
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# Data
# ============================================================
species_label = ['Carp', 'Crab', 'Crayfish', 'Mullet', 'Snakehead']
species_en    = ['carp', 'crab', 'crayfish', 'ms', 'wuli']
colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5']

# Convergence epochs
det_epochs = [2, 2, 48, 1, 1]
kp_epochs  = [2, 4, 53, 3, 2]

# Load loss curves (first 100 iterations from scalars)
loss_curves = {}
for sp in species_en:
    rundirs = sorted(ROOT.glob(f'runs/det_{sp}_*'))
    if not rundirs:
        continue
    subdirs = [d for d in rundirs[-1].iterdir() if d.is_dir() and d.name.startswith('20')]
    if not subdirs:
        continue
    scalar_file = subdirs[0] / "vis_data" / "scalars.json"
    if not scalar_file.exists():
        continue
    with open(scalar_file) as f:
        lines = f.readlines()
    losses = []
    for line in lines[:150]:
        d = json.loads(line)
        if 'loss' in d and 'coco/bbox_mAP' not in d:
            losses.append(d['loss'])
    if losses:
        loss_curves[sp] = losses[:100]

# ============================================================
# Figure: Two-panel plot
# (a) Training loss curves (first 100 iters)
# (b) Epochs to convergence
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel (a): Detection loss curves
ax = axes[0]
for sp, c in zip(species_en, colors):
    if sp in loss_curves:
        curve = loss_curves[sp]
        x = np.arange(len(curve))
        ax.plot(x, curve, color=c, linewidth=1.3, alpha=0.85, label=sp.capitalize())
ax.set_xlabel('Iteration', fontsize=11)
ax.set_ylabel('Detection Loss', fontsize=11)
ax.set_title('(a) Initial Convergence (first 100 iterations)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=2, loc='upper right')
ax.set_xlim(0, 99)
ax.grid(alpha=0.25)

# Panel (b): Epochs to best metric
ax = axes[1]
x = np.arange(len(species_label))
w = 0.35
bars_det = ax.bar(x - w/2, det_epochs, w, label='Detection', color='#4472C4', edgecolor='white')
bars_kp  = ax.bar(x + w/2, kp_epochs, w, label='Keypoint', color='#ED7D31', edgecolor='white')
for bar, val in zip(bars_det, det_epochs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val),
            ha='center', va='bottom', fontsize=8, fontweight='bold', color='#4472C4')
for bar, val in zip(bars_kp, kp_epochs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val),
            ha='center', va='bottom', fontsize=8, fontweight='bold', color='#ED7D31')
ax.set_ylabel('Epochs to Best', fontsize=11)
ax.set_title('(b) Convergence Speed', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(species_label, fontsize=10)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.25)

plt.tight_layout(pad=2)
out_path = ROOT / "evals" / "training_efficiency.png"
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"[SAVED] {out_path}")
plt.close(fig)

# Also copy to paper/images
import shutil
dst = ROOT.parent / "paper" / "images" / "training_efficiency.png"
shutil.copy(out_path, dst)
print(f"[COPIED] {dst}")

print("\n[DONE]")
