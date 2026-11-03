"""Recover DSCMR Pascal row -> image mapping with the ORIGINAL (Oxford Caffe) VGG19 weights.

Weights: Keras port of the Oxford Caffe VGG19 (fchollet/deep-learning-models v0.1,
vgg19_weights_tf_dim_ordering_tf_kernels.h5, sha256 c872d82e...), stored outside the repo in
D:/Caches/vgg19_caffe. Caffe preprocessing: BGR, 0-255, minus ImageNet BGR mean.
fc7 (Keras 'fc2') after ReLU is compared by cosine to DSCMR train/test image rows.
Step 1 (--stage extract): features for all 1000 UIUC images under several crop variants.
Step 2 (--stage match): cosine diagnostics + Hungarian matching; writes DSCMR/data_ids/*.
"""
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import torch
import torchvision
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
H5 = Path(os.environ.get('VGG19_CAFFE_H5', 'D:/Caches/vgg19_caffe/vgg19_keras_caffe.h5'))  # override per machine
IMG = ROOT/'CLIP4CMR/data/pascal_uiuc_seed42/images'
UP = ROOT/'DSCMR/upstream/data/pascal'
OUT = ROOT/'DSCMR/data_ids'
MEAN_BGR = np.array([103.939, 116.779, 123.68], np.float32)
CONV_IDX = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]


def build():
    net = torchvision.models.vgg19(weights=None)
    with h5py.File(H5, 'r') as f:
        names = [n for n in f.keys() if 'conv' in n]
        names.sort(key=lambda n: (int(n[5]), int(n.split('conv')[1])))
        for i, n in zip(CONV_IDX, names):
            k, b = _wb(f[n])
            net.features[i].weight.data = torch.from_numpy(k.transpose(3, 2, 0, 1).copy())
            net.features[i].bias.data = torch.from_numpy(b)
        for i, n in ((0, 'fc1'), (3, 'fc2'), (6, 'predictions')):
            w, b = _wb(f[n])
            if n == 'fc1':  # keras flattens HWC, torch CHW
                w = w.reshape(7, 7, 512, -1).transpose(2, 0, 1, 3).reshape(25088, -1)
            net.classifier[i].weight.data = torch.from_numpy(w.T.copy()); net.classifier[i].bias.data = torch.from_numpy(b)
    return net.eval()


def _wb(g):
    leaves = []
    g.visititems(lambda name, obj: leaves.append((name, np.array(obj))) if isinstance(obj, h5py.Dataset) else None)
    w = [a for n, a in leaves if 'kernel' in n or n.endswith('W') or 'W_' in n]
    b = [a for n, a in leaves if 'bias' in n or n.endswith('b') or 'b_' in n]
    if not w:  # fallback: larger array is the kernel
        leaves.sort(key=lambda t: -t[1].ndim); w, b = [leaves[0][1]], [leaves[1][1]]
    return w[0].astype(np.float32), b[0].astype(np.float32)


def prep(im, mode):
    im = im.convert('RGB')
    if mode == 'warp224':
        im = im.resize((224, 224), Image.BILINEAR); crops = [np.asarray(im, np.float32)]
    else:  # short side 256, then center crop or 10-crop mean
        w, h = im.size; s = 256/min(w, h); im = im.resize((round(w*s), round(h*s)), Image.BILINEAR)
        a = np.asarray(im, np.float32); H, W = a.shape[:2]; c = [((H-224)//2, (W-224)//2)]
        if mode == 'tencrop256':
            c += [(0, 0), (0, W-224), (H-224, 0), (H-224, W-224)]
        crops = [a[y:y+224, x:x+224] for y, x in c]
        if mode == 'tencrop256':
            crops += [x[:, ::-1] for x in crops]
    return torch.stack([torch.from_numpy((x[..., ::-1] - MEAN_BGR).transpose(2, 0, 1).copy()) for x in crops])


def extract():
    net = build().cuda()
    ids = [str(i) for i in np.load(ROOT/'CLIP4CMR/data/encoders/dfn5b_h14.npz', allow_pickle=True)['ids']]
    fc7 = torch.nn.Sequential(*list(net.classifier.children())[:5])  # fc6 relu drop fc7 relu
    out = {}
    with torch.inference_mode():
        x = prep(Image.open(IMG/f'{ids[0]}.jpg'), 'warp224').cuda()
        top = net(x).softmax(1).topk(3); print('sanity', ids[0], top.indices.tolist(), top.values.tolist())
        for mode in ('warp224', 'center256', 'tencrop256'):
            feats = []
            for i in ids:
                x = prep(Image.open(IMG/f'{i}.jpg'), mode).cuda()
                f = fc7(torch.flatten(net.avgpool(net.features(x)), 1)).mean(0)
                feats.append(f.cpu().numpy())
            out[mode] = np.stack(feats); print('extracted', mode, flush=True)
    np.savez_compressed(OUT/'caffe_vgg19_fc7.npz', ids=np.array(ids), **out)


def mat(n):
    m = sio.loadmat(UP/f'{n}.mat'); return m[[k for k in m if not k.startswith('_')][0]]


def l2n(x):
    x = np.asarray(x, np.float64); return x/np.linalg.norm(x, axis=1, keepdims=True)


def match():
    z = np.load(OUT/'caffe_vgg19_fc7.npz')
    ids = z['ids']; lab = np.load(ROOT/'CLIP4CMR/data/encoders/dfn5b_h14.npz', allow_pickle=True)['labels']
    D = {s: (mat(f'{s}_img'), mat(f'{s}_img_lab').ravel()) for s in ('train', 'test')}
    report = {}
    for mode in [k for k in z.files if k != 'ids']:
        ours = l2n(z[mode]); r = {}
        for s, (X, y) in D.items():
            C = l2n(X) @ ours.T  # rows x 1000
            best = C.max(1); arg = C.argmax(1); srt = np.sort(C, 1)
            r[s] = {'best_cos_pct': np.percentile(best, [0, 5, 50, 95]).round(4).tolist(),
                    'margin_med': float(np.median(srt[:, -1]-srt[:, -2])),
                    'unique_best': int(len(set(arg.tolist())))}
        report[mode] = r; print(mode, json.dumps(r), flush=True)
    mode = max(report, key=lambda m: report[m]['train']['best_cos_pct'][2])
    ours = l2n(z[mode]); allX = np.vstack([D['train'][0], D['test'][0]]); ally = np.concatenate([D['train'][1], D['test'][1]])
    C = l2n(allX) @ ours.T
    row, col = linear_sum_assignment(-C)
    cos = C[row, col]
    # label mapping implied by the matching (DSCMR label -> our alphabetical label)
    conf = np.zeros((20, 20), int)
    for i, j in zip(row, col): conf[ally[i], lab[j]] += 1
    lmap = {int(a): int(conf[a].argmax()) for a in range(20)}
    pure = float(sum(conf[a].max() for a in range(20))/len(row))
    rows = [{'split': 'train' if i < 800 else 'test', 'row': int(i if i < 800 else i-800), 'dscmr_label': int(ally[i]),
             'image_id': str(ids[j]), 'our_label': int(lab[j]), 'cos': float(c)} for i, j, c in zip(row, col, cos)]
    res = {'mode': mode, 'diagnostics': report, 'label_map_dscmr_to_alpha': lmap, 'label_purity': pure,
           'cos_pct_0_1_5_50': np.percentile(cos, [0, 1, 5, 50]).round(4).tolist(),
           'n_cos_ge_0.99': int((cos >= .99).sum()), 'n_cos_ge_0.95': int((cos >= .95).sum()), 'n_rows': len(rows),
           'weights_sha256': 'c872d82e40328910cd7a12763dae043d56eb00efe6a137f31899514d637d4cdb'}
    (OUT/'recovered_ids_caffe.json').write_text(json.dumps({'summary': res, 'rows': rows}, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'diagnostics'}, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=('extract', 'match'), required=True)
    a = p.parse_args(); torch.cuda.set_per_process_memory_fraction(0.25)
    extract() if a.stage == 'extract' else match()
