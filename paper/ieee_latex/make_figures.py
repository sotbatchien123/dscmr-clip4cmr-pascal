"""Figures of the paper, computed from the saved test embeddings (same source as the notebook).

    python paper/ieee_latex/make_figures.py        (from the repository root)

Writes fig_results.pdf next to this file (per-class AP is plotted in the notebook).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.metrics import label_relevance, map_all  # noqa: E402

OUT = Path(__file__).resolve().parent
V3 = ROOT/'reports/improve_v3_20261030'
SEEDS = range(42, 47)
CTR = json.loads((ROOT/'reports/revision_20261112/stats/results.json').read_text(encoding='utf-8'))['baselines']
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 7.5, 'pdf.fonttype': 42})
C1, C2 = '#1F4E79', '#9DC3E6'
RUNS = {'c4_base': V3/'confirm/base_rn50_c4/seed{s}', 'c4_AB_rn50': ROOT/'reports/clip4cmr_author_split_20261030/confirm/rn50/A_sem+B_seed{s}',
        'c4_frozen': V3/'confirm/frozen_dfn5b_c4AB/seed{s}', 'c4_ft_base': V3/'finetune/confirm/c4_base_L2/seed{s}',
        'c4_ft': V3/'finetune/confirm/c4_AB_L2/seed{s}', 'ds_base': V3/'confirm/base_rn50_ds/seed{s}',
        'ds_ens': V3/'confirm/ens3_ds_capmean/seed{s}', 'ds_ft': V3/'finetune/confirm/ds_L4/seed{s}'}


def load(key):
    out = []
    for s in SEEDS:
        t = np.load(Path(str(RUNS[key]).format(s=s))/'test_features.npz')
        sc = t['image'] @ t['text'].T; rel = label_relevance(t['label'], t['label'])
        out.append((t, sc, rel, map_all(sc, rel)['mAP@ALL']*100, map_all(sc.T, rel.T)['mAP@ALL']*100))
    return out


R = {k: load(k) for k in RUNS}


def stats(k):
    return [(np.mean([r[i] for r in R[k]]), np.std([r[i] for r in R[k]], ddof=1)) for i in (3, 4)]


def results_figure():
    c4 = [('Base\n(RN50)', 'c4_base'), ('+A+B\n(RN50)', 'c4_AB_rn50'), ('+DFN5B\n(frozen)', 'c4_frozen'),
          ('FT 2 bl.,\norig. head', 'c4_ft_base'), ('FT 2 bl.\n+A+B', 'c4_ft')]
    ds = [('RN50', 'ds_base'), ('Ensemble\n3 enc.', 'ds_ens'), ('DFN5B\nFT 4 bl.', 'ds_ft')]
    ctr = CTR['DFN5B']['classify-then-rank (zero-shot clf)']; ctr = (ctr['I2T']+ctr['T2I'])/2
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.05), gridspec_kw={'width_ratios': [5, 3]})
    for ax, data, title in ((axes[0], c4, '(a) CLIP4CMR'), (axes[1], ds, '(b) DSCMR head')):
        x = np.arange(len(data)); w = .38; st = [stats(k) for _, k in data]
        for j, col in enumerate((C1, C2)):
            ax.bar(x+(j-.5)*w, [d[j][0] for d in st], w, yerr=[d[j][1] for d in st], capsize=1.3,
                   error_kw={'lw': .6}, color=col, label=('I2T', 'T2I')[j])
        for i, d in enumerate(st):
            top = max(m+s for m, s in d)
            ax.text(i, top+.9, f'{(d[0][0]+d[1][0])/2:.1f}', ha='center', va='bottom', fontsize=6.1,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .4}, zorder=4)
        ax.axhline(ctr, ls=':', lw=.9, color='#555555', label='classify-rank\n(DFN5B, no head)')
        ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data], fontsize=5.2)
        ax.set_ylim(64, 80); ax.set_yticks([65, 70, 75, 80]); ax.tick_params(labelsize=6.5); ax.set_title(title, fontsize=7.5); ax.grid(axis='y', lw=.3, alpha=.5); ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].axhline(69.5, ls='--', lw=.7, color='#C00000', label='published [2]')
    h = dict(zip(*axes[0].get_legend_handles_labels()[::-1]))  # label -> handle
    axes[0].legend([h['published [2]'], h['classify-rank\n(DFN5B, no head)']], ['published [2]', 'classify-rank\n(DFN5B, no head)'],
                   fontsize=5.2, frameon=False, loc='upper left', handlelength=1.6)
    axes[1].legend([h['I2T'], h['T2I']], ['I2T', 'T2I'], fontsize=5.6, frameon=False, loc='upper left')
    axes[0].set_ylabel('mAP@ALL (%)')
    fig.tight_layout(pad=.3, w_pad=.6); fig.savefig(OUT/'fig_results.pdf'); plt.close(fig)


if __name__ == '__main__':
    results_figure(); print('written', OUT/'fig_results.pdf')
