"""Figure for the CLIP4CMR improvement study (needs matplotlib). Usage: python -m scripts.clip4cmr.plot_improve [out_dir]"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]/'reports/clip4cmr_improve_20261030'
agg = json.loads((out/'aggregate.json').read_text(encoding='utf-8'))['aggregate']
variants = ['base', 'B', 'A_res', 'A_sem', 'A_sem+B']
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (key, title) in zip(axes, (('rn50', 'RN50'), ('vitl14', 'ViT-L/14'))):
    for k, dr in enumerate(('I2T', 'T2I')):
        m = [agg[f'{key}/{v}'][dr]['mAP@ALL'][0]*100 for v in variants]
        s = [agg[f'{key}/{v}'][dr]['mAP@ALL'][1]*100 for v in variants]
        ax.bar(np.arange(len(variants))+(k-.5)*.38, m, .38, yerr=s, capsize=3, label=dr)
    zs = agg[f'{key}/zero_shot']['capmean']
    ax.axhline((zs['I2T']+zs['T2I'])*50, color='gray', ls='--', lw=1, label='zero-shot CLIP (TB 2 chiều)')
    ax.set_xticks(range(len(variants)), variants); ax.set_title(f'CLIP4CMR — {title}'); ax.grid(axis='y', alpha=.3)
axes[0].set_ylabel('Test mAP@ALL (%), TB ± SD 5 seed'); axes[0].set_ylim(60, 76)  # axis starts at 60%: shows differences
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, fontsize=9, frameon=False)
fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(out/'test_map.png', dpi=150)
print('saved', out/'test_map.png')
