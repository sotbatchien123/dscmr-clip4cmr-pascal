"""Synthetic software checks only; these are NOT benchmark experiments."""
import unittest
import numpy as np
from common.metrics import recall_at_k, map_all, hamming_scores, label_relevance, cosine_scores


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

    def test_perfect_ranking_gives_map_one(self):
        # Moi ket qua dung deu dung dau moi truy van -> R@1 = 1 va mAP = 1
        s = np.array([[.9, .8, .1, .2], [.1, .2, .9, .8]])
        r = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=bool)
        res = map_all(s, r)
        self.assertAlmostEqual(res['mAP@ALL'], 1.)
        self.assertEqual(res['queries_without_positive'], 0)
        self.assertEqual(recall_at_k(s, r)['R@1'], 1.)

    def test_single_positive_ranked_last(self):
        # Chi 1 ket qua dung va no dung cuoi trong 4 ung vien: AP = 1/4
        self.assertAlmostEqual(map_all([[4, 3, 2, 1]], [[0, 0, 0, 1]])['mAP@ALL'], .25)

    def test_label_relevance_1d_class_ids(self):
        # Nhan lop 1 chieu: [0, 1] so voi [0, 0, 1]
        r = label_relevance([0, 1], [0, 0, 1])
        expected = np.array([[True, True, False], [False, False, True]])
        np.testing.assert_array_equal(r, expected)

    def test_recall_k_larger_than_gallery(self):
        # K=10 lon hon so ung vien (4): ham van chay, ket qua dung
        s = np.array([[4, 3, 2, 1]])
        self.assertEqual(recall_at_k(s, [[0, 0, 0, 1]], ks=(1, 10)),
                         {'R@1': 0., 'R@10': 1.})

    def test_cosine_zero_vector_raises(self):
        # Vector toan so 0 khong chuan hoa duoc -> phai bao ValueError
        with self.assertRaises(ValueError):
            cosine_scores([[0, 0, 0]], [[1, 0, 0]])

    def test_class_labels_to_cosine_pipeline_perfect(self):
        # Luong day du nhan lop -> do lien quan -> cosine -> chi so, ca 2 chieu
        labels = np.array([0, 0, 1, 1, 2, 2])
        emb = np.eye(3, dtype=np.float32)[labels]
        r = label_relevance(labels, labels)
        s = cosine_scores(emb, emb)
        self.assertAlmostEqual(map_all(s, r)['mAP@ALL'], 1.)
        self.assertAlmostEqual(map_all(s.T, r.T)['mAP@ALL'], 1.)
        self.assertEqual(recall_at_k(s, r)['R@1'], 1.)


if __name__ == '__main__':
    unittest.main()
