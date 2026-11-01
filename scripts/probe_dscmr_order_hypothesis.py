"""Test whether DSCMR's Pascal rows follow the canonical UIUC page order (class, then filename).

DSCMR labels: per class 40 train rows then 5 test rows (class blocks in a shuffled order).
Hypothesis H: within each class, rows = images sorted by filename; train = first 40,
test = some later block of 5 (positions 40-44 or 45-49).
Evidence: representational similarity (Spearman of within-class Gram matrices) between the
DSCMR features (VGG19 fc7 / sentence-CNN) and our DFN5B features, under H vs random mappings.
No labels or test results of any model are used. Output: reports/probe_dscmr_order_20261031.json
"""
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT/'DSCMR/upstream/data/pascal'
rng = np.random.default_rng(0)


def mat(name):
    m = sio.loadmat(UP/f'{name}.mat'); return m[[k for k in m if not k.startswith('_')][0]]


def l2n(x):
    x = np.asarray(x, np.float64); return x/np.linalg.norm(x, axis=1, keepdims=True)


def rsa(a, b):
    """Spearman correlation of upper triangles of cosine Gram matrices."""
    iu = np.triu_indices(len(a), 1)
    return float(spearmanr((l2n(a) @ l2n(a).T)[iu], (l2n(b) @ l2n(b).T)[iu])[0])


ds = {'train_img': mat('train_img'), 'train_txt': mat('train_txt'), 'test_img': mat('test_img'), 'test_txt': mat('test_txt'),
      'train_y': mat('train_img_lab').ravel(), 'test_y': mat('test_img_lab').ravel()}
e = np.load(ROOT/'CLIP4CMR/data/encoders/dfn5b_h14.npz', allow_pickle=True)
ids, lab = np.array([str(i) for i in e['ids']]), e['labels']
ours = {'img': e['image'], 'txt': e['text_author']}

out = {'class_order_train': [int(c) for c in dict.fromkeys(ds['train_y'])]}
# 1) label mapping: class-mean RSA (DSCMR label k vs our alphabetical label k)
cm = lambda X, y: np.stack([X[y == c].mean(0) for c in range(20)])
out['class_mean_rsa_img'] = rsa(cm(ds['train_img'], ds['train_y']), cm(ours['img'], lab))
out['class_mean_rsa_txt'] = rsa(cm(ds['train_txt'], ds['train_y']), cm(ours['txt'], lab))

# 2) within-class order hypotheses
res = {}
for split, n, starts in (('train', 40, [0]), ('test', 5, [40, 45])):
    for mod in ('img', 'txt'):
        for st in starts:
            hyp, null = [], []
            for c in range(20):
                theirs = ds[f'{split}_{mod}'][ds[f'{split}_y'] == c]
                cls = np.flatnonzero(lab == c)[np.argsort(ids[lab == c])]  # filename order
                hyp.append(rsa(theirs, ours[mod][cls[st:st+n]]))
                if n >= 10:
                    null += [rsa(theirs, ours[mod][rng.permutation(cls)[:n]]) for _ in range(20)]
            res[f'{split}_{mod}_start{st}'] = {'hyp_mean': float(np.mean(hyp)), 'hyp_per_class': np.round(hyp, 3).tolist(),
                                              **({'null_mean': float(np.mean(null)), 'null_sd': float(np.std(null))} if null else {})}
out['within_class'] = res

# 3) test block: direct nearest-neighbour check is impossible (different feature spaces); instead, for each class,
#    score every 5-subset start position 0..45 of the filename order by RSA with DSCMR test rows (img+txt)
pos = {}
for c in range(20):
    cls = np.flatnonzero(lab == c)[np.argsort(ids[lab == c])]
    sc = [np.mean([rsa(ds[f'test_{m}'][ds['test_y'] == c], ours[m][cls[s:s+5]]) for m in ('img', 'txt')]) for s in range(46)]
    pos[c] = int(np.argmax(sc))
out['test_best_start_per_class'] = pos
(ROOT/'reports/probe_dscmr_order_20261031.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != 'within_class'}, indent=0))
for k, v in res.items():
    print(k, 'hyp %.3f' % v['hyp_mean'], ('null %.3f±%.3f' % (v['null_mean'], v['null_sd'])) if 'null_mean' in v else '')
