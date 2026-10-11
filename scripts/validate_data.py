import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import karpathy_manifest, sha256


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='kind', required=True)
    f = sub.add_parser('flickr30k')
    f.add_argument('--annotations', required=True)
    f.add_argument('--images', required=True)
    f.add_argument('--split', default='test')
    d = sub.add_parser('dscmr')
    d.add_argument('--root', required=True)
    args = p.parse_args()
    if args.kind == 'flickr30k':
        images, captions, owners = karpathy_manifest(args.annotations, args.images, args.split)
        counts = [owners.count(i) for i in range(len(images))]
        result = {'images': len(images), 'captions': len(captions), 'split': args.split,
                  'captions_per_image_min': min(counts), 'captions_per_image_max': max(counts),
                  'annotations_sha256': sha256(args.annotations),
                  'standard_1k_5captions_shape': len(images) == 1000 and min(counts) == max(counts) == 5,
                  'note': 'Shape alone does not prove official split identity.'}
    else:
        import numpy as np
        from scipy.io import loadmat
        root = Path(args.root)
        keys = ['train_img', 'test_img', 'train_txt', 'test_txt', 'train_img_lab', 'test_img_lab']
        values = {}
        result = {'files': {}, 'protocol_verified': False}
        for key in keys:
            path = root / (key + '.mat')
            value = loadmat(path)[key]
            if not np.isfinite(value).all():
                raise ValueError(f'Nonfinite values in {key}')
            values[key] = value
            result['files'][key] = {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': sha256(path)}
        for split in ['train', 'test']:
            if len(values[split+'_img']) != len(values[split+'_txt']):
                raise ValueError(f'{split} modalities have different row counts')
            if values[split+'_img_lab'].shape != (len(values[split+'_img']), 1):
                raise ValueError('Upstream loader expects N x 1 class IDs')
        if values['train_img'].shape[1] != values['test_img'].shape[1] or values['train_txt'].shape[1] != values['test_txt'].shape[1]:
            raise ValueError('Train/test feature dimensions differ')
        for key in ['train_img_lab', 'test_img_lab']:
            if not np.equal(values[key], np.floor(values[key])).all():
                raise ValueError('Labels must be integer IDs')
        if not set(values['test_img_lab'].ravel()).issubset(set(values['train_img_lab'].ravel())):
            raise ValueError('Test has classes not represented in train')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
