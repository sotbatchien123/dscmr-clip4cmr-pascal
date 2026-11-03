"""Figures for DSCMR Assignment 3 round 2. Needs matplotlib; reads JSON outputs only.

Usage: python -m scripts.dscmr.plot_loss_tuning <round2_out_dir> <round1_out_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out, round1 = Path(sys.argv[1]), Path(sys.argv[2])
tuning = json.loads((out/'tuning.json').read_text(encoding='utf-8'))
agg = json.loads((out/'aggregate.json').read_text(encoding='utf-8'))['aggregate']

# 1. Tuning heatmaps: CV mean validation mAP.
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, name, xkey, ykey, xl, yl in ((axes[0], 'B0_original', 2, 1, 'β (cặp ảnh–văn bản)', 'α (không gian chung)'),
                                     (axes[1], 'B1_supcon', 'tau', 1, 'τ (temperature)', 'λ (SupCon)')):
    rows = [r for r in tuning['rows'] if r['config']['name'] == name]
    xs = sorted({r['config']['tau'] if xkey == 'tau' else r['config']['weights'][xkey] for r in rows})
    ys = sorted({r['config']['weights'][ykey] for r in rows})
    grid = np.full((len(ys), len(xs)), np.nan)
    for r in rows:
        xv = r['config']['tau'] if xkey == 'tau' else r['config']['weights'][xkey]
        grid[ys.index(r['config']['weights'][ykey]), xs.index(xv)] = r['mean']*100
    im = ax.imshow(grid, cmap='viridis', origin='lower')
    for i in range(len(ys)):
        for j in range(len(xs)):
            ax.text(j, i, f'{grid[i, j]:.2f}', ha='center', va='center', color='white', fontsize=9)
    ax.set_xticks(range(len(xs)), [f'{v:g}' for v in xs]); ax.set_yticks(range(len(ys)), [f'{v:g}' for v in ys])
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(f'{name}: CV mAP (%)')
    fig.colorbar(im, ax=ax, shrink=.8)
fig.tight_layout(); fig.savefig(out/'tuning_heatmaps.png', dpi=150); plt.close(fig)

# 2. Test mAP, round 2 confirmation.
names = list(agg)
fig, ax = plt.subplots(figsize=(8, 4))
w = .38
for k, d in enumerate(('I2T', 'T2I')):
    m = [agg[n][d]['mAP@ALL']['mean']*100 for n in names]
    s = [agg[n][d]['mAP@ALL']['std']*100 for n in names]
    ax.bar(np.arange(len(names))+(k-.5)*w, m, w, yerr=s, capsize=4, label=d)
lo = min(agg[n][d]['mAP@ALL']['mean'] for n in names for d in ('I2T', 'T2I'))*100
ax.set_ylim(lo-4, lo+8); ax.set_xticks(range(len(names)), names)
ax.set_ylabel('Test mAP@ALL (%)'); ax.set_title('Vòng 2 (đã tuning): trung bình ± SD qua 5 seed')
ax.legend(); ax.grid(axis='y', alpha=.3)
fig.tight_layout(); fig.savefig(out/'test_map_tuned.png', dpi=150); plt.close(fig)

# 3. Share of each loss term until the best epoch: round 1 vs round 2 (seed 42).
def share(history, best):
    rows = [np.asarray(h['raw_losses'])*np.asarray(h['weights']) for h in history[:best]]
    return np.mean([r/r.sum() for r in rows], 0)*100
bars = []
for v in ('B0_original', 'B1_supcon'):
    r = json.loads((round1/'main'/f'{v}_seed42'/'result.json').read_text())
    bars.append((f'{v}\nvòng 1', share(json.loads((round1/'main'/f'{v}_seed42'/'history.json').read_text()), r['best_epoch'])))
for v in ('B0_original', 'B1_supcon'):
    r = json.loads((out/'confirm'/f'{v}_seed42'/'result.json').read_text())
    bars.append((f'{v}*\nvòng 2', np.asarray(r['mean_term_share_until_best'])*100))
fig, ax = plt.subplots(figsize=(8, 4))
bottom = np.zeros(len(bars))
for k, lab in enumerate(('Loss nhãn', 'Không gian chung', 'Cặp ảnh–văn bản')):
    vals = np.array([b[1][k] for b in bars])
    ax.bar(range(len(bars)), vals, bottom=bottom, label=lab)
    for i, (v, b0) in enumerate(zip(vals, bottom)):
        if v > 4: ax.text(i, b0+v/2, f'{v:.1f}%', ha='center', va='center', fontsize=8, color='white')
    bottom += vals
ax.set_xticks(range(len(bars)), [b[0] for b in bars]); ax.set_ylabel('Tỷ trọng trong tổng loss (%)')
ax.set_title('Tỷ trọng thành phần loss tới epoch tốt nhất (seed 42)'); ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
fig.tight_layout(); fig.savefig(out/'loss_share.png', dpi=150); plt.close(fig)
print('saved', [p.name for p in out.glob('*.png')])
