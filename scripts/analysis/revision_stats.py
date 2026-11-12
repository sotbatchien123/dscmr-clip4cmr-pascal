"""Simple baselines and query-level bootstrap intervals (revision experiments).

1. Baselines on frozen features (RN50 author split, DFN5B ViT-H/14), no retrieval head:
   - zero-shot retrieval: cosine between image and text (concatenated captions / caption mean);
   - classify-then-rank, zero-shot classifier: p = softmax(100 * cos(f, prompt_c)), score = p_img . p_txt;
   - classify-then-rank, linear probe: one multinomial logistic regression per modality on the 800
     training items (C picked by 3-fold CV on train), score = p_img . p_txt.
   Classification accuracy on the 200 test items is reported for every classifier.
2. Nearest-prototype accuracy of the trained CLIP4CMR heads that use the fixed prompt prototypes.
3. Bootstrap over test items (both directions of an item are resampled together, gallery fixed),
   per-query AP averaged over the five seeds: 95% interval of mAP and of paired differences.

    python -m scripts.analysis.revision_stats
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from common.metrics import label_relevance, map_all
from scripts.clip4cmr.improve import stratified_folds

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT/'reports/improve_v3_20261030'
SEEDS = range(42, 47)
FEATURES = {'RN50': ROOT/'CLIP4CMR/data/pascal_author_split_mirror/rn50.npz',
            'DFN5B': ROOT/'CLIP4CMR/data/author_split_v3/dfn5b_h14.npz'}
MODELS = {  # saved test embeddings, five seeds each
    'C4 base RN50': V3/'confirm/base_rn50_c4/seed{s}/test_features.npz',
    'C4 A+B RN50': ROOT/'reports/clip4cmr_author_split_20261030/confirm/rn50/A_sem+B_seed{s}/test_features.npz',
    'C4 A+B DFN5B frozen': V3/'confirm/frozen_dfn5b_c4AB/seed{s}/test_features.npz',
    'C4 base DFN5B FT2': V3/'finetune/confirm/c4_base_L2/seed{s}/test_features.npz',
    'C4 A+B DFN5B FT2': V3/'finetune/confirm/c4_AB_L2/seed{s}/test_features.npz',
    'DSCMR RN50': V3/'confirm/base_rn50_ds/seed{s}/test_features.npz',
    'DSCMR ens3': V3/'confirm/ens3_ds_capmean/seed{s}/test_features.npz',
    'DSCMR DFN5B FT4': V3/'finetune/confirm/ds_L4/seed{s}/test_features.npz'}
PAIRS = [('C4 base RN50', 'C4 A+B RN50'), ('C4 base RN50', 'C4 A+B DFN5B frozen'),
         ('C4 A+B DFN5B frozen', 'C4 A+B DFN5B FT2'), ('C4 base DFN5B FT2', 'C4 A+B DFN5B FT2'),
         ('C4 base RN50', 'C4 A+B DFN5B FT2'), ('C4 base RN50', 'C4 base DFN5B FT2'), ('DSCMR RN50', 'DSCMR DFN5B FT4'),
         ('DSCMR ens3', 'DSCMR DFN5B FT4'), ('DSCMR DFN5B FT4', 'C4 A+B DFN5B FT2'),
         ('CtR zero-shot RN50', 'C4 base RN50'), ('CtR zero-shot RN50', 'C4 A+B RN50'),
         ('CtR zero-shot DFN5B', 'C4 A+B DFN5B frozen'), ('CtR zero-shot DFN5B', 'C4 A+B DFN5B FT2'),
         ('CtR probe DFN5B', 'C4 A+B DFN5B FT2'), ('CtR zero-shot DFN5B', 'DSCMR DFN5B FT4')]
B = 10000


def l2n(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def ap_per_query(s, rel):
    order = np.argsort(-s, axis=1, kind='stable')
    r = np.take_along_axis(rel, order, axis=1)
    prec = np.cumsum(r, axis=1) / np.arange(1, s.shape[1]+1)
    return (prec*r).sum(1) / np.maximum(r.sum(1), 1)


def per_item(s, y):
    rel = label_relevance(y, y)
    return np.stack([ap_per_query(s, rel), ap_per_query(s.T, rel.T)], 1)


def both(s, y):
    rel = label_relevance(y, y)
    return {'I2T': map_all(s, rel)['mAP@ALL']*100, 'T2I': map_all(s.T, rel.T)['mAP@ALL']*100}


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def probe(xtr, ytr, xte, folds):
    best = None
    for C in (0.1, 0.3, 1, 3, 10, 30, 100):
        acc = []
        for f in range(len(folds)):
            fit = np.concatenate([folds[g] for g in range(len(folds)) if g != f])
            m = LogisticRegression(C=C, max_iter=5000).fit(xtr[fit], ytr[fit])
            acc.append(m.score(xtr[folds[f]], ytr[folds[f]]))
        if best is None or np.mean(acc) > best[0] + 1e-12:
            best = (np.mean(acc), C)
    m = LogisticRegression(C=best[1], max_iter=5000).fit(xtr, ytr)
    return m.predict_proba(xte), best[1]


def baselines():
    """Returns the metrics and per-query AP (items, 2) of the classify-then-rank baselines."""
    out, q = {}, {}
    for name, path in FEATURES.items():
        d = np.load(path); tr, te, y = d['train_index'], d['test_index'], d['labels']
        capmean = np.stack([d['captions'][d['caption_owner'] == i].mean(0) for i in range(len(y))])
        img, txt, P = l2n(d['image']), l2n(capmean), l2n(d['prompts'])
        r = {'zero-shot (concat)': both(img[te] @ l2n(d['text_concat'])[te].T, y[te]),
             'zero-shot (caption mean)': both(img[te] @ txt[te].T, y[te])}
        pi, pt = softmax(100*img[te] @ P.T), softmax(100*txt[te] @ P.T)
        q[f'CtR zero-shot {name}'] = per_item(pi @ pt.T, y[te])
        r['classify-then-rank (zero-shot clf)'] = {**both(pi @ pt.T, y[te]), 'acc_image': (pi.argmax(1) == y[te]).mean()*100,
                                                   'acc_text': (pt.argmax(1) == y[te]).mean()*100}
        folds = stratified_folds(y[tr])
        pi, ci_ = probe(img[tr], y[tr], img[te], folds); pt, ct = probe(txt[tr], y[tr], txt[te], folds)
        q[f'CtR probe {name}'] = per_item(pi @ pt.T, y[te])
        r['classify-then-rank (linear probe)'] = {**both(pi @ pt.T, y[te]), 'acc_image': (pi.argmax(1) == y[te]).mean()*100,
                                                  'acc_text': (pt.argmax(1) == y[te]).mean()*100, 'C': [ci_, ct]}
        out[name] = r
    return out, q


def model_queries():
    """Per-query AP (I2T and T2I for each of the 200 items), averaged over seeds; plus per-seed mAP."""
    q, maps, acc = {}, {}, {}
    prompts = l2n(np.load(FEATURES['DFN5B'])['prompts'])
    for name, pat in MODELS.items():
        if not all(Path(str(pat).format(s=s)).exists() for s in SEEDS):
            print('skip (missing runs):', name); continue
        aps, mm, ac = [], [], []
        for s in SEEDS:
            t = np.load(str(pat).format(s=s)); sc = t['image'] @ t['text'].T; rel = label_relevance(t['label'], t['label'])
            a = np.stack([ap_per_query(sc, rel), ap_per_query(sc.T, rel.T)], 1)  # (items, 2)
            aps.append(a); mm.append(a.mean()*100)
            if 'A+B DFN5B' in name:  # fixed prompt prototypes in DFN5B space
                ac.append([((l2n(t[k]) @ prompts.T).argmax(1) == t['label']).mean()*100 for k in ('image', 'text')])
        q[name] = np.mean(aps, 0); maps[name] = (float(np.mean(mm)), float(np.std(mm, ddof=1)))
        if ac:
            acc[name] = np.mean(ac, 0).tolist()
    return q, maps, acc


def bootstrap(q):
    rng = np.random.default_rng(0); n = next(iter(q.values())).shape[0]
    idx = rng.integers(0, n, size=(B, n))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    single = {k: {'mean': float(v.mean()*100), 'ci95': ci(v.mean(1)[idx].mean(1)*100)} for k, v in q.items()}
    paired = {}
    for a, b in PAIRS:
        if a not in q or b not in q:
            continue
        diff = (q[b] - q[a]).mean(1); boot = diff[idx].mean(1)*100
        paired[f'{b} - {a}'] = {'diff': float(diff.mean()*100), 'ci95': ci(boot), 'p_le_0': float((boot <= 0).mean())}
    return single, paired


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT/'reports/revision_20261112/stats')
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    base, bq = baselines()
    q, maps, acc = model_queries()
    q.update(bq); maps.update({k: (float(v.mean()*100), 0.) for k, v in bq.items()})
    single, paired = bootstrap(q)
    res = {'baselines': base, 'models_mean_sd': maps, 'prototype_accuracy': acc, 'bootstrap_single': single,
           'bootstrap_paired': paired, 'bootstrap': f'{B} resamples of the 200 test items, seed 0'}
    (args.out/'results.json').write_text(json.dumps(res, indent=2, default=float), encoding='utf-8')
    L = ['## Baselines (frozen features, test 200)', '', '| Features | Method | I2T | T2I | Mean | Acc img | Acc txt |', '|---|---|---:|---:|---:|---:|---:|']
    for f, r in base.items():
        for m, v in r.items():
            L.append(f"| {f} | {m} | {v['I2T']:.2f} | {v['T2I']:.2f} | {(v['I2T']+v['T2I'])/2:.2f} | "
                     f"{v.get('acc_image', float('nan')):.1f} | {v.get('acc_text', float('nan')):.1f} |")
    L += ['', '## Models: mean mAP over seeds, 95% bootstrap interval over test items', '', '| Model | mAP ± SD (seeds) | 95% CI | Acc img / txt |', '|---|---:|---|---|']
    for k in q:
        a = acc.get(k); L.append(f"| {k} | {maps[k][0]:.2f} ± {maps[k][1]:.2f} | [{single[k]['ci95'][0]:.2f}, {single[k]['ci95'][1]:.2f}] | "
                                 + (f'{a[0]:.1f} / {a[1]:.1f}' if a else '') + ' |')
    L += ['', '## Paired differences (points)', '', '| Comparison | Diff | 95% CI | P(boot <= 0) |', '|---|---:|---|---:|']
    for k, v in paired.items():
        L.append(f"| {k} | {v['diff']:+.2f} | [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] | {v['p_le_0']:.4f} |")
    (args.out/'REPORT.md').write_text('\n'.join(L)+'\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
