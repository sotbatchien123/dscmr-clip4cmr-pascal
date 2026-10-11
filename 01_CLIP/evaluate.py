"""Flickr30K full-gallery evaluation with the official OpenAI CLIP package.
Custom evaluator; no claim of paper reproduction until protocol is audited.
"""
import argparse
import datetime
import json
import random
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from common.data import karpathy_manifest, sha256
from common.metrics import recall_at_k


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--annotations', required=True)
    p.add_argument('--images', required=True)
    p.add_argument('--model', default='ViT-B/32')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--limit-images', type=int, default=None)
    p.add_argument('--truncate', action='store_true')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if a.batch_size < 1 or (a.limit_images is not None and a.limit_images < 1):
        p.error('batch-size and limit-images must be positive')
    import torch
    import clip
    from PIL import Image
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu explicitly if needed')
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a fresh output directory to preserve evidence')
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    images, captions, owners = karpathy_manifest(a.annotations, a.images, a.split, a.limit_images)
    model, preprocess = clip.load(a.model, device=a.device, jit=False)
    model.eval()
    image_vectors, text_vectors = [], []
    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(images), a.batch_size):
            batch = []
            for filename in images[i:i+a.batch_size]:
                with Image.open(filename) as im:
                    batch.append(preprocess(im.convert('RGB')))
            v = model.encode_image(torch.stack(batch).to(a.device)).float()
            v = v / v.norm(dim=-1, keepdim=True)
            image_vectors.append(v.cpu().numpy())
            print(f'images {min(i+a.batch_size, len(images))}/{len(images)}', flush=True)
        for i in range(0, len(captions), a.batch_size):
            tokens = clip.tokenize(captions[i:i+a.batch_size], truncate=a.truncate).to(a.device)
            v = model.encode_text(tokens).float()
            v = v / v.norm(dim=-1, keepdim=True)
            text_vectors.append(v.cpu().numpy())
    iv, tv = np.concatenate(image_vectors), np.concatenate(text_vectors)
    scores = iv @ tv.T
    relevance = np.arange(len(images))[:, None] == np.asarray(owners)[None, :]
    metrics = {'I2T': recall_at_k(scores, relevance), 'T2I': recall_at_k(scores.T, relevance.T)}
    result = {'method': 'CLIP', 'status': 'evaluated', 'metrics_unit': 'fraction_0_1',
              'mode': 'zero-shot', 'metrics': metrics, 'model': a.model,
              'dataset': 'Flickr30K', 'images': len(images), 'captions': len(captions),
              'scope': 'subset_smoke_test' if a.limit_images else 'requested_split',
              'split': a.split, 'annotation_sha256': sha256(a.annotations),
              'is_reproduction_verified': False,
              'protocol_note': 'Match checkpoint/backbone, split IDs, caption preprocessing and paper row before comparison.',
              'elapsed_seconds_inference_and_metrics': time.perf_counter()-start,
              'timestamp': datetime.datetime.now().astimezone().isoformat(),
              'python': sys.version, 'torch': torch.__version__, 'clip_source': clip.__file__,
              'device': a.device, 'args': vars(a)}
    (out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(out / 'embeddings.npz', image=iv, text=tv, caption_owner=np.asarray(owners))
    (out / 'gallery.json').write_text(json.dumps({'images': images, 'captions': captions, 'caption_owner': owners}, ensure_ascii=False), encoding='utf-8')
    examples = []
    for qi in range(min(20, len(captions))):
        ranked = np.argsort(-scores[:, qi], kind='stable')[:10]
        examples.append({'query': captions[qi], 'ground_truth': images[owners[qi]],
                         'top10': [images[k] for k in ranked], 'correct_at_1': bool(ranked[0] == owners[qi])})
    (out / 'examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
