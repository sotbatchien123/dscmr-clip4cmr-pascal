"""Retrieval metrics. Scores: rows=query, columns=gallery; larger is better.

R@K is query hit rate (at least one relevant result), not document recall.
AP@ALL includes queries with no positives as zero by default. Tie breaking is
stable gallery order, which must be reported when comparing hashing methods.
"""
import numpy as np


def _inputs(scores, relevant):
    s = np.asarray(scores)
    r = np.asarray(relevant, dtype=bool)
    if s.ndim != 2 or s.shape != r.shape or min(s.shape) == 0:
        raise ValueError('Expected nonempty score and relevance matrices of equal shape')
    if not np.isfinite(s).all():
        raise ValueError('Scores contain NaN/Inf')
    return s, r


def recall_at_k(scores, relevant, ks=(1, 5, 10)):
    s, r = _inputs(scores, relevant)
    if not r.any(axis=1).all():
        raise ValueError('Caption benchmark has queries with no ground-truth item')
    order = np.argsort(-s, axis=1, kind='stable')
    ranked = np.take_along_axis(r, order, axis=1)
    if any(k < 1 for k in ks):
        raise ValueError('K must be positive')
    return {f'R@{k}': float(ranked[:, :k].any(axis=1).mean()) for k in ks}


def map_all(scores, relevant, zero_positive='include'):
    s, r = _inputs(scores, relevant)
    order = np.argsort(-s, axis=1, kind='stable')
    ranked = np.take_along_axis(r, order, axis=1)
    positives = ranked.sum(axis=1)
    precision = np.cumsum(ranked, axis=1) / np.arange(1, s.shape[1] + 1)
    ap = (precision * ranked).sum(axis=1) / np.maximum(positives, 1)
    if zero_positive == 'exclude':
        ap = ap[positives > 0]
        if not len(ap):
            raise ValueError('No query has a positive')
    elif zero_positive != 'include':
        raise ValueError('zero_positive must be include or exclude')
    return {'mAP@ALL': float(ap.mean()), 'queries_without_positive': int((positives == 0).sum())}


def label_relevance(query_labels, gallery_labels):
    q, g = np.asarray(query_labels), np.asarray(gallery_labels)
    if q.ndim == g.ndim == 1:
        return q[:, None] == g[None, :]
    if q.ndim == g.ndim == 2 and q.shape[1] == g.shape[1]:
        return (q.astype(np.int64) @ g.astype(np.int64).T) > 0
    raise ValueError('Use class IDs or consistent binary multi-hot labels')


def cosine_scores(q, g):
    q, g = np.asarray(q, dtype=np.float32), np.asarray(g, dtype=np.float32)
    if q.ndim != 2 or g.ndim != 2 or q.shape[1] != g.shape[1]:
        raise ValueError('Embedding dimensions differ')
    qn, gn = np.linalg.norm(q, axis=1, keepdims=True), np.linalg.norm(g, axis=1, keepdims=True)
    if np.any(qn == 0) or np.any(gn == 0):
        raise ValueError('Zero embedding')
    return (q / qn) @ (g / gn).T


def hamming_scores(q, g):
    q, g = np.asarray(q), np.asarray(g)
    if q.ndim != 2 or g.ndim != 2 or q.shape[1] != g.shape[1]:
        raise ValueError('Hash dimensions differ')
    values = set(np.unique(np.concatenate([q.ravel(), g.ravel()])).tolist())
    if values.issubset({0, 1}):
        q, g = 2 * q.astype(np.int64) - 1, 2 * g.astype(np.int64) - 1
    elif values.issubset({-1, 1}):
        q, g = q.astype(np.int64), g.astype(np.int64)
    else:
        raise ValueError('Hashes must be binary 0/1 or -1/+1; zero logits need explicit handling')
    return -(q.shape[1] - q @ g.T) / 2
