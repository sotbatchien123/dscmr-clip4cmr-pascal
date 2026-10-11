import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def karpathy_manifest(annotation, image_root, split='test', limit=None):
    root = Path(image_root).resolve()
    raw = json.loads(Path(annotation).read_text(encoding='utf-8'))
    images, captions, owners = [], [], []
    seen = set()
    for item in raw['images']:
        if item['split'] != split:
            continue
        path = (root / item.get('filepath', '') / item['filename']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Image path escapes dataset root')
        if path in seen:
            raise ValueError(f'Duplicate image: {path}')
        if not path.is_file():
            raise FileNotFoundError(path)
        texts = [s.get('raw', ' '.join(s.get('tokens', []))) for s in item['sentences']]
        if not texts or any(not x.strip() for x in texts):
            raise ValueError(f'Empty captions: {path}')
        owners.extend([len(images)] * len(texts))
        captions.extend(texts)
        images.append(str(path))
        seen.add(path)
        if limit is not None and len(images) >= limit:
            break
    if not images:
        raise ValueError(f'No images in split {split}')
    return images, captions, owners
