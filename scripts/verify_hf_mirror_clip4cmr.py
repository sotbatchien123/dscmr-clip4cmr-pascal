"""Verify the third-party HF mirror of CLIP4CMR Pascal features against local RN50 extraction.

Loads clip_train/test.pkl with an unpickler that only allows numpy globals, then checks
shapes/labels/ids, split overlap with the local seed-42 split, and whether image/text
features match local RN50 features (image; text_concat; caption mean; single captions).
Usage: python -I verify_hf_mirror_clip4cmr.py   (writes JSON next to the mirror files)
"""
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MIRROR = ROOT/'CLIP4CMR/data/hf_mirror_lcmi'
LOCAL = ROOT/'CLIP4CMR/data/pascal_uiuc_seed42_multicaption/rn50.npz'
ALLOWED = {('numpy', 'dtype'), ('numpy', 'ndarray'), ('numpy.core.multiarray', '_reconstruct')}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f'blocked {module}.{name}')
        return getattr(__import__(module, fromlist=[name]), name)


def load(name):
    with open(MIRROR/name, 'rb') as f:
        return SafeUnpickler(f).load()


def l2n(x):
    x = x.astype(np.float64)
    return x/np.linalg.norm(x, axis=1, keepdims=True)


def best_cos(a, b):
    """mean and min of cosine between matched rows."""
    c = (l2n(a)*l2n(b)).sum(1)
    return float(c.mean()), float(c.min())


tr, te = load('clip_train.pkl'), load('clip_test.pkl')
loc = np.load(LOCAL, allow_pickle=False)
lid = [str(x) for x in loc['ids']]
pos = {k: i for i, k in enumerate(lid)}
report = {'mirror_files': {}, 'split': {}, 'features': {}}
for nm, d in (('train', tr), ('test', te)):
    report['mirror_files'][nm] = {k: [list(v.shape), str(v.dtype)] for k, v in d.items()}
mids = {nm: [str(x) for x in d['ids']] for nm, d in (('train', tr), ('test', te))}
local_train = {lid[i] for i in loc['train_index']}
local_test = {lid[i] for i in loc['test_index']}
report['split'] = {
    'mirror_train_n': len(mids['train']), 'mirror_test_n': len(mids['test']),
    'unique_ids_total': len(set(mids['train']) | set(mids['test'])),
    'train_test_overlap': len(set(mids['train']) & set(mids['test'])),
    'ids_missing_from_uiuc': sorted((set(mids['train']) | set(mids['test'])) - set(lid)),
    'mirror_test_in_local_test': len(set(mids['test']) & local_test),
    'mirror_test_in_local_train': len(set(mids['test']) & local_train),
    'test_per_class': np.bincount(te['label'].astype(int)).tolist(),
}
# label agreement with UIUC class (local labels index CLASS order)
all_ids = mids['train'] + mids['test']
all_lab = np.concatenate([tr['label'], te['label']]).astype(int)
idx = np.array([pos[i] for i in all_ids if i in pos])
ok = [i in pos for i in all_ids]
report['split']['label_match_local'] = float((loc['labels'][idx] == all_lab[ok]).mean())

img = np.concatenate([tr['image'], te['image']])[ok]
txt = np.concatenate([tr['text'], te['text']])[ok]
li = loc['image'][idx]
report['features']['image_vs_local_rn50 (mean,min cos)'] = best_cos(img, li)
report['features']['image_max_abs_diff'] = float(np.abs(img.astype(np.float32)-li.astype(np.float32)).max())
report['features']['text_vs_local_concat'] = best_cos(txt, loc['text_concat'][idx])
caps, owner = loc['captions'], loc['caption_owner']
capmean = np.stack([caps[owner == i].astype(np.float64).mean(0) for i in range(len(lid))])
report['features']['text_vs_local_caption_mean'] = best_cos(txt, capmean[idx])
# best single caption per row (does mirror text = one caption?)
best = []
for r, i in enumerate(idx):
    c = l2n(caps[owner == i]) @ l2n(txt[r:r+1]).T
    best.append(float(c.max()))
best = np.array(best)
report['features']['text_vs_best_single_caption (mean,min)'] = [float(best.mean()), float(best.min())]
report['features']['rows_with_single_caption_cos>0.999'] = int((best > 0.999).sum())
out = MIRROR/'verification.json'
out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(report, indent=2, ensure_ascii=False))
