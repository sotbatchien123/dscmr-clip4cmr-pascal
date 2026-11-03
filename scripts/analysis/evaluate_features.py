"""Optional independent check on exported features, NOT a substitute protocol.
NPZ keys: query, gallery, query_labels, gallery_labels.
"""
import argparse
import json
import numpy as np
from common.metrics import cosine_scores, hamming_scores, label_relevance, map_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--representation', choices=['real', 'binary'], required=True)
    p.add_argument('--zero-positive', choices=['include', 'exclude'], default='include')
    a = p.parse_args()
    z = np.load(a.input, allow_pickle=False)
    scoring = cosine_scores if a.representation == 'real' else hamming_scores
    scores = scoring(z['query'], z['gallery'])
    relevance = label_relevance(z['query_labels'], z['gallery_labels'])
    result = map_all(scores, relevance, a.zero_positive)
    result.update(tie_break='stable_gallery_order', metrics_unit='fraction_0_1', source=a.input,
                  protocol='independent_AP_at_ALL_label_overlap', paper_comparability_verified=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
