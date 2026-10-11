"""Synthetic software checks only; these are NOT benchmark experiments."""
import unittest
import numpy as np
from common.metrics import recall_at_k, map_all, hamming_scores, label_relevance


class RetrievalMetricsTests(unittest.TestCase):
    def test_multiple_captions_and_two_directions(self):
        s = np.array([[.9, .8, .1, .2], [.1, .2, .9, .8]])
        r = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=bool)
        self.assertEqual(recall_at_k(s, r)['R@1'], 1.)
        self.assertEqual(recall_at_k(s.T, r.T)['R@1'], 1.)

    def test_ap_hand_computed(self):
        # Positives at ranks 2 and 4: (1/2 + 2/4)/2 = .5
        self.assertAlmostEqual(map_all([[4, 3, 2, 1]], [[0, 1, 0, 1]])['mAP@ALL'], .5)

    def test_zero_positive_policies(self):
        self.assertEqual(map_all([[1, 0], [1, 0]], [[1, 0], [0, 0]])['mAP@ALL'], .5)
        self.assertEqual(map_all([[1, 0], [1, 0]], [[1, 0], [0, 0]], 'exclude')['mAP@ALL'], 1.)
        with self.assertRaises(ValueError):
            recall_at_k([[1, 0]], [[0, 0]])

    def test_stable_hash_ties(self):
        s = hamming_scores([[1, 0]], [[0, 0], [1, 1]])
        np.testing.assert_array_equal(s, [[-1, -1]])
        self.assertEqual(map_all(s, [[0, 1]])['mAP@ALL'], .5)

    def test_multi_label_overlap(self):
        r = label_relevance([[1, 0, 1]], [[0, 1, 1], [0, 1, 0]])
        np.testing.assert_array_equal(r, [[True, False]])

    def test_wide_hash_does_not_overflow(self):
        q = np.ones((1, 512), dtype=np.int8)
        self.assertEqual(hamming_scores(q, -q)[0, 0], -512)

    def test_invalid_scores(self):
        with self.assertRaises(ValueError):
            map_all([[float('nan')]], [[1]])


if __name__ == '__main__':
    unittest.main()
