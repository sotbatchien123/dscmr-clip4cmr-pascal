"""Do the RN50 conclusions depend on the mirror's precomputed features? (revision experiment)

The third-party mirror ships RN50 features for the author split; 78 of its 1000 image features
differ from our own RN50 extraction of the same UIUC images (cos < 0.999). This script
1. lists those images (class, split, cosine) and compares the mirror and local vectors;
2. rebuilds the author split from our own RN50 extraction (same ids, same 800/200 split, own
   image features, own caption/prompt features; the concatenated text follows the mirror rule
   only for the mirror, so here the local concatenation is used);
3. reruns the CLIP4CMR baseline and A_sem+B (alpha 0.3) for seeds 42-46 under the same protocol
   and opens the test once at the end.

    python -m scripts.clip4cmr.self_features_check
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.clip4cmr import improve as ci
from scripts.baseline import train_baseline as ts
from common.paths import from_repo

ROOT = ci.ROOT
MIRROR = ROOT/'CLIP4CMR/data/pascal_author_split_mirror/rn50.npz'
LOCAL = ROOT/'CLIP4CMR/data/pascal_uiuc_seed42_multicaption/rn50.npz'
DATA = ROOT/'CLIP4CMR/data/author_split_self'
VARIANTS = ('base', 'A_sem+B')


def l2n(x):
    return x/np.linalg.norm(x, axis=1, keepdims=True)


def build():
    """Local RN50 features reordered to the mirror's rows (labels/split taken from the mirror)."""
    m, loc = np.load(MIRROR), np.load(LOCAL)
    pos = {str(k): i for i, k in enumerate(loc['ids'])}
    idx = np.array([pos[str(i)] for i in m['ids']])
    new_of_old = {int(o): n for n, o in enumerate(idx)}
    owner = np.array([new_of_old[int(o)] for o in loc['caption_owner']]); order = np.argsort(owner, kind='stable')
    d = {'image': loc['image'][idx], 'text_concat': loc['text_concat'][idx], 'captions': loc['captions'][order],
         'caption_owner': owner[order], 'prompts': loc['prompts'], 'labels': m['labels'], 'ids': m['ids'],
         'train_index': m['train_index'], 'test_index': m['test_index']}
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez(DATA/'rn50.npz', **d)
    (DATA/'provenance.json').write_text(json.dumps({'source': 'own RN50 extraction of UIUC images, rows/split of the '
                                                    'author-split mirror'}, indent=2), encoding='utf-8')
    cos_img = (l2n(m['image'].astype(np.float64))*l2n(d['image'].astype(np.float64))).sum(1)
    cos_txt = (l2n(m['text_concat'].astype(np.float64))*l2n(d['text_concat'].astype(np.float64))).sum(1)
    bad = np.flatnonzero(cos_img < .999); test = set(m['test_index'].tolist())
    return {'images_cos_below_0.999': int(len(bad)), 'in_test': int(sum(i in test for i in bad)),
            'cos_quantiles_of_mismatches': np.percentile(cos_img[bad], [0, 25, 50, 75, 100]).round(4).tolist(),
            'classes_of_mismatches': np.bincount(m['labels'][bad], minlength=21).tolist(),
            'text_concat_cos_below_0.999': int((cos_txt < .999).sum()),
            'mismatched_ids': [str(m['ids'][i]) for i in bad]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT/'reports/revision_20261112/self_features')
    p.add_argument('--device', default='cuda')
    args = p.parse_args(); args.out = args.out.resolve(); args.out.mkdir(parents=True, exist_ok=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    audit = build(); ci.write_json(args.out/'mismatch_audit.json', audit)
    print({k: v for k, v in audit.items() if k != 'mismatched_ids'}, flush=True)
    ci.set_feats(DATA); d = ci.load_backbone('rn50', args.device); pan = ts.prototype_loss()
    tr = d['train']; inner, val = ci.fixed_split(d['labels'][tr])
    runs = [(v, s, ci.train_run(args, d, pan, tr[inner], tr[val], v, .3, s, args.out/f'{v}_seed{s}', ci.CONFIRM_EPOCHS))
            for v in VARIANTS for s in ci.SEEDS]
    res = []
    for v, s, r in runs:  # test opened only now
        mode, use_b = ci.VARIANTS[v]
        ts.seed_all(s); model = ci.build_model(d, mode, .3, args.device)
        model.load_state_dict(torch.load(from_repo(r['run'])/'best.pt', map_location=args.device)['state_dict'])
        m, fi, ft = ci.evaluate(model, d['image'][d['test']], ci.eval_text(d, d['test'], use_b), d['labels'][d['test']])
        np.savez_compressed(from_repo(r['run'])/'test_features.npz', image=fi, text=ft, label=d['labels'][d['test']])
        res.append({'variant': v, 'seed': s, 'I2T': m['I2T']['mAP@ALL'], 'T2I': m['T2I']['mAP@ALL']})
    ci.write_json(args.out/'results.json', res)
    for v in VARIANTS:
        x = [(r['I2T']+r['T2I'])*50 for r in res if r['variant'] == v]
        print(f'{v}: {np.mean(x):.2f} +- {np.std(x, ddof=1):.2f}', flush=True)
    gain = [((a['I2T']+a['T2I'])-(b['I2T']+b['T2I']))*50 for a, b in zip(res[5:], res[:5])]
    print('paired gain A_sem+B - base:', np.round(gain, 2), f'mean {np.mean(gain):+.2f}')


if __name__ == '__main__':
    main()
