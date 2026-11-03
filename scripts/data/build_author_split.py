"""Build CLIP4CMR feature files on the author's Pascal 800/200 split (from the HF mirror pkl).

rn50  : image/text_concat = mirror features exactly (author pipeline, fp16 -> fp32);
        captions/prompts = local per-caption RN50 embeddings re-ordered to mirror rows (for B / A_sem).
vitl14: image/captions/prompts = local ViT-L/14 features re-ordered; text_concat re-encoded with
        the author's text rule (5 captions joined, punctuation and digits removed, truncate 77).
Rows: mirror train (800) then mirror test (200); labels = mirror label - 1 (author labels are 1..20).
Output: CLIP4CMR/data/pascal_author_split_mirror/{rn50,vitl14}.npz + provenance.json
"""
import hashlib
import json
import re
import string
from pathlib import Path

import clip
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
from scripts.data.verify_hf_mirror import MIRROR, load  # noqa: E402

LOCAL = ROOT/'CLIP4CMR/data/pascal_uiuc_seed42_multicaption'
OUT = ROOT/'CLIP4CMR/data/pascal_author_split_mirror'


def author_text(caps):
    t = re.sub('[{}]'.format(string.punctuation), '', ' '.join(caps))
    return re.sub(r'[\d]', '', t)


def main():
    tr, te = load('clip_train.pkl'), load('clip_test.pkl')
    ids = [str(x) for x in np.concatenate([tr['ids'], te['ids']])]
    labels = np.concatenate([tr['label'], te['label']]).astype(np.int64) - 1
    records = {r['id']: r for r in json.loads(
        (ROOT/'reports/clip4cmr_investigation_20261014/pascal_raw_manifest.json').read_text(encoding='utf-8'))}
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {'source': 'https://huggingface.co/spaces/shivangibithel/LCMI_T2I (main), third-party mirror, not author-confirmed',
            'mirror_sha256': {f: hashlib.sha256((MIRROR/f).read_bytes()).hexdigest() for f in ('clip_train.pkl', 'clip_test.pkl')},
            'train_ids': ids[:800], 'test_ids': ids[800:], 'labels': 'mirror label - 1',
            'verification': json.loads((MIRROR/'verification.json').read_text(encoding='utf-8'))['split'],
            'text_rule_probe': json.loads((MIRROR/'text_rule_probe.json').read_text(encoding='utf-8')), 'backbones': {}}
    for key, name in (('rn50', 'RN50'), ('vitl14', 'ViT-L/14')):
        loc = np.load(LOCAL/f'{key}.npz')
        lid = [str(x) for x in loc['ids']]; pos = {k: i for i, k in enumerate(lid)}
        idx = np.array([pos[i] for i in ids])
        assert (loc['labels'][idx] == labels).all(), 'label mismatch with UIUC classes'
        new_of_old = {int(o): n for n, o in enumerate(idx)}
        keep = np.array([o in new_of_old for o in loc['caption_owner']])
        order = np.argsort([new_of_old[int(o)] for o in loc['caption_owner']], kind='stable')
        caps, owner = loc['captions'][order], np.array([new_of_old[int(o)] for o in loc['caption_owner']])[order]
        assert keep.all()
        if key == 'rn50':
            image = np.concatenate([tr['image'], te['image']]).astype(np.float32)
            concat = np.concatenate([tr['text'], te['text']]).astype(np.float32)
            how = 'mirror features (author extraction)'
        else:
            image = loc['image'][idx]
            model, _ = clip.load(name, device='cuda', download_root=str(ROOT/'checkpoints')); model.eval()
            texts = [author_text(records[i]['captions']) for i in ids]
            with torch.inference_mode():
                concat = np.concatenate([model.encode_text(clip.tokenize(texts[s:s+50], truncate=True).cuda()).float().cpu().numpy()
                                         for s in range(0, len(texts), 50)])
            del model; torch.cuda.empty_cache()
            how = 'local ViT-L/14 image; text_concat re-encoded with author text rule'
        np.savez(OUT/f'{key}.npz', image=image, text_concat=concat, captions=caps, caption_owner=owner,
                 prompts=loc['prompts'], labels=labels, ids=np.array(ids),
                 train_index=np.arange(800), test_index=np.arange(800, 1000))
        meta['backbones'][key] = {'model': name, 'features': how, 'dim': int(image.shape[1])}
        print('saved', key, image.shape, caps.shape, flush=True)
    (OUT/'provenance.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
