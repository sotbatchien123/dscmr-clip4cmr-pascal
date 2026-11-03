"""Frozen CLIP features for the CLIP4CMR improvement study (A: semantic prototypes, B: per-caption).

For each backbone (RN50, ViT-L/14) and all 1000 local UIUC Pascal images, stores:
  image        (1000, D)  encode_image, same preprocessing as scripts/data/build_pascal_local.py
  text_concat  (1000, D)  all captions joined, truncate=True (the existing baseline rule)
  captions     (M, D)     every caption encoded separately; caption_owner (M,) row index
  prompts      (20, D)    "a photo of a {class name}." for semantic prototypes (method A)
plus ids/labels in manifest order and the existing 800/200 split from provenance.json.
Images must already exist locally (downloaded by scripts/data/build_pascal_local.py).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import clip
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'CLIP4CMR/data/pascal_uiuc_seed42'
OUT = ROOT/'CLIP4CMR/data/pascal_uiuc_seed42_multicaption'
BACKBONES = {'RN50': 'rn50', 'ViT-L/14': 'vitl14'}
PROMPT = 'a photo of a {}.'
CLASS_NAMES = {'diningtable': 'dining table', 'pottedplant': 'potted plant', 'tvmonitor': 'tv monitor'}


def main():
    records = json.loads((ROOT/'reports/clip4cmr_investigation_20261014/pascal_raw_manifest.json').read_text(encoding='utf-8'))
    prov = json.loads((DATA/'provenance.json').read_text(encoding='utf-8'))
    ids = np.array([r['id'] for r in records])
    classes = prov['class_names']
    labels = np.array([classes.index(r['class']) for r in records])
    pos = {v: i for i, v in enumerate(ids)}
    train = np.array([pos[i] for i in prov['train_ids']]); test = np.array([pos[i] for i in prov['test_ids']])
    assert len(train) == 800 and len(test) == 200 and not set(train) & set(test)
    owner = np.array([i for i, r in enumerate(records) for _ in r['captions']])
    caps = [c for r in records for c in r['captions']]
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {'prompt_template': PROMPT, 'class_names': classes,
            'prompt_class_names': [CLASS_NAMES.get(c, c) for c in classes],
            'captions_per_image': {str(k): int(v) for k, v in zip(*np.unique(np.bincount(owner), return_counts=True))},
            'split': 'existing local 800/200 split from pascal_uiuc_seed42/provenance.json', 'backbones': {}}
    for name, key in BACKBONES.items():
        model, preprocess = clip.load(name, device='cuda', download_root=str(ROOT/'checkpoints'))
        model.eval()
        enc_t = lambda texts, trunc=False: model.encode_text(clip.tokenize(texts, truncate=trunc).cuda()).float().cpu().numpy()
        image, concat, capf = [], [], []
        with torch.inference_mode():
            for s in range(0, len(records), 32):
                batch = records[s:s+32]
                pixels = torch.stack([preprocess(Image.open(DATA/'images'/f"{r['id']}.jpg").convert('RGB')) for r in batch]).cuda()
                image.append(model.encode_image(pixels).float().cpu().numpy())
                concat.append(enc_t([' '.join(r['captions']) for r in batch], True))
            for s in range(0, len(caps), 128):
                capf.append(enc_t(caps[s:s+128]))  # single captions never exceed 77 tokens; error if they do
            prompts = enc_t([PROMPT.format(n) for n in meta['prompt_class_names']])
        arrays = {'image': np.concatenate(image), 'text_concat': np.concatenate(concat), 'captions': np.concatenate(capf),
                  'caption_owner': owner, 'prompts': prompts, 'labels': labels, 'ids': ids,
                  'train_index': train, 'test_index': test}
        assert all(np.isfinite(v).all() for k, v in arrays.items() if v.dtype.kind == 'f')
        np.savez(OUT/f'{key}.npz', **arrays)
        ckpt = ROOT/'checkpoints'/(Path(clip.clip._MODELS[name]).name)
        meta['backbones'][name] = {'file': f'{key}.npz', 'dim': int(arrays['image'].shape[1]),
                                   'checkpoint': ckpt.name, 'checkpoint_sha256': hashlib.sha256(ckpt.read_bytes()).hexdigest()}
        print('saved', key, arrays['image'].shape, arrays['captions'].shape, flush=True)
        del model; torch.cuda.empty_cache()
    (OUT/'provenance.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
