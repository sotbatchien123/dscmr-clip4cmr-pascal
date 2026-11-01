"""Extract frozen features for strong open_clip encoders on all 1000 Pascal (UIUC) pairs.

Per encoder -> CLIP4CMR/data/encoders/<tag>.npz (rows in pascal_raw_manifest order):
  image (1000,D); text_author (1000,D) = author text rule (5 captions joined, punctuation and digits
  removed, tokenizer truncation); captions (M,D) + caption_owner; prompts (20,D) "a photo of a {class}.";
  ids, labels (UIUC class index, same order as CLIP4CMR provenance class_names).
fp16 inference, small batches, VRAM capped (shared GPU). Resumable: skips existing npz.
Usage: python extract_encoders.py [tag ...]
"""
import json
import os
import re
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'.feat_libs'))
os.environ.setdefault('HF_HOME', str(ROOT/'checkpoints/hf'))

import numpy as np  # noqa: E402
import open_clip  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

ENCODERS = {  # tag: (open_clip model, pretrained)
    'vitl14_336': ('ViT-L-14-336', 'openai'),
    'eva02l_336': ('EVA02-L-14-336', 'merged2b_s6b_b61k'),
    'siglip_so400m_384': ('ViT-SO400M-14-SigLIP-384', 'webli'),
    'dfn5b_h14': ('ViT-H-14-quickgelu', 'dfn5b'),
}
OUT = ROOT/'CLIP4CMR/data/encoders'
IMAGES = ROOT/'CLIP4CMR/data/pascal_uiuc_seed42/images'
CLASS_NAMES = {'diningtable': 'dining table', 'pottedplant': 'potted plant', 'tvmonitor': 'tv monitor'}


def author_text(caps):
    t = re.sub('[{}]'.format(string.punctuation), '', ' '.join(caps))
    return re.sub(r'[\d]', '', t)


def main():
    tags = sys.argv[1:] or list(ENCODERS)
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('VRAM_FRACTION', '0.40')))
    records = json.loads((ROOT/'reports/clip4cmr_investigation_20261014/pascal_raw_manifest.json').read_text(encoding='utf-8'))
    classes = json.loads((ROOT/'CLIP4CMR/data/pascal_uiuc_seed42/provenance.json').read_text(encoding='utf-8'))['class_names']
    labels = np.array([classes.index(r['class']) for r in records])
    owner = np.array([i for i, r in enumerate(records) for _ in r['captions']])
    caps = [c for r in records for c in r['captions']]
    prompts = [f'a photo of a {CLASS_NAMES.get(c, c)}.' for c in classes]
    OUT.mkdir(parents=True, exist_ok=True)
    for tag in tags:
        if (OUT/f'{tag}.npz').exists():
            print('skip', tag, flush=True); continue
        name, pre = ENCODERS[tag]
        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pre, device='cpu')
        model = model.half().cuda()  # cast on CPU first: an fp32 copy of ViT-H does not fit the shared GPU
        tok = open_clip.get_tokenizer(name); model.eval()
        enc_t = lambda texts: np.concatenate([model.encode_text(tok(texts[s:s+32]).cuda()).float().cpu().numpy()
                                              for s in range(0, len(texts), 32)])
        with torch.inference_mode():
            image = []
            for s in range(0, len(records), 8):
                px = torch.stack([preprocess(Image.open(IMAGES/f"{r['id']}.jpg").convert('RGB')) for r in records[s:s+8]]).cuda().half()
                image.append(model.encode_image(px).float().cpu().numpy())
            arrays = {'image': np.concatenate(image), 'text_author': enc_t([author_text(r['captions']) for r in records]),
                      'captions': enc_t(caps), 'prompts': enc_t(prompts)}
        np.savez(OUT/f'{tag}.npz', **arrays, caption_owner=owner, labels=labels, ids=np.array([r['id'] for r in records]))
        meta_p = OUT/'provenance.json'
        meta = json.loads(meta_p.read_text(encoding='utf-8')) if meta_p.exists() else {}
        meta[tag] = {'open_clip_model': name, 'pretrained': pre, 'dim': int(arrays['image'].shape[1]), 'precision': 'fp16',
                     'open_clip_version': open_clip.__version__, 'prompt': 'a photo of a {class}.'}
        meta_p.write_text(json.dumps(meta, indent=2), encoding='utf-8')
        print('saved', tag, arrays['image'].shape, torch.cuda.max_memory_allocated()/2**30, 'GB peak', flush=True)
        del model; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


if __name__ == '__main__':
    main()
