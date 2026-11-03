"""Find which text-preprocessing rule reproduces the HF-mirror CLIP4CMR text features.

Candidate rules follow upstream load_original_data (first 50 words, strip punctuation and
digits) applied to the UIUC captions in different orders/joins. Prints, per rule, the mean
cosine to mirror text features and how many rows match (cos > 0.999).
Usage: python -m scripts.data.probe_mirror_text_rule
"""
import json
import re
import string
from pathlib import Path

import clip
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
from scripts.data.verify_hf_mirror import load, l2n  # noqa: E402

records = {r['id']: r for r in json.loads(
    (ROOT/'reports/clip4cmr_investigation_20261014/pascal_raw_manifest.json').read_text(encoding='utf-8'))}
tr, te = load('clip_train.pkl'), load('clip_test.pkl')
ids = [str(x) for x in np.concatenate([tr['ids'], te['ids']])]
target = l2n(np.concatenate([tr['text'], te['text']]))
P = string.punctuation


def upstream(t, n=50):
    t = ' '.join(t.split(' ')[:n])
    t = re.sub('[{}]'.format(P), '', t)
    return re.sub(r'[\d]', '', t)


RULES = {
    'join_space_raw_trunc': lambda c: ' '.join(c),
    'upstream50_join_space': lambda c: upstream(' '.join(c)),
    'upstream50_join_newline': lambda c: upstream('\n'.join(c)),
    'upstream50_join_nospace': lambda c: upstream(''.join(c)),
    'upstream50_sorted': lambda c: upstream(' '.join(sorted(c))),
    'upstream_nolimit_join_space': lambda c: upstream(' '.join(c), 10**6),
    'upstream50_first_caption': lambda c: upstream(c[0]),
}
model, _ = clip.load('RN50', device='cuda', download_root=str(ROOT/'checkpoints'))
model.eval()
out = {}
with torch.inference_mode():
    for name, rule in RULES.items():
        texts = [rule(records[i]['captions']) for i in ids]
        feats = np.concatenate([model.encode_text(clip.tokenize(texts[s:s+100], truncate=True).cuda()).float().cpu().numpy()
                                for s in range(0, len(texts), 100)])
        c = (l2n(feats)*target).sum(1)
        out[name] = {'mean_cos': round(float(c.mean()), 4), 'n_cos>0.999': int((c > 0.999).sum()),
                     'n_cos>0.99': int((c > 0.99).sum())}
        print(name, out[name], flush=True)
(ROOT/'CLIP4CMR/data/hf_mirror_lcmi/text_rule_probe.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
