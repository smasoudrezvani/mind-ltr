import numpy as np

from mind_ltr.ranking.metrics import dcg_at_k, mrr, ndcg_at_k, recall_at_k
from mind_ltr.rerank.mmr import mmr_rerank
from mind_ltr.rerank.position_bias import ipw_sample_weights


def test_perfect_ranking_has_ndcg_1():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.2, 0.1])  # relevant items scored highest
    assert ndcg_at_k(y, s, k=4) == 1.0


def test_worst_ranking_has_low_ndcg():
    y = np.array([1, 0, 0, 0])
    s = np.array([0.1, 0.9, 0.8, 0.7])  # relevant item ranked last
    assert ndcg_at_k(y, s, k=4) < 0.5


def test_ndcg_no_relevant_is_zero():
    y = np.zeros(5)
    s = np.random.rand(5)
    assert ndcg_at_k(y, s) == 0.0


def test_dcg_position_discount():
    """Same single click is worth more at rank 1 than rank 2."""
    y_click_first = np.array([1, 0])
    y_click_second = np.array([0, 1])
    s = np.array([0.9, 0.1])
    assert dcg_at_k(y_click_first, s, 2) > dcg_at_k(y_click_second, s, 2)


def test_mrr():
    y = np.array([0, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1])  # first relevant at rank 3
    assert mrr(y, s) == 1.0 / 3.0


def test_recall_at_k():
    y = np.array([1, 1, 0, 0, 1])
    s = np.array([0.9, 0.1, 0.8, 0.7, 0.2])  # only 1 of 3 relevant in top-2
    assert recall_at_k(y, s, k=2) == 1.0 / 3.0


def test_ipw_upweights_low_positions():
    clicks = np.array([1, 1, 0])
    positions = np.array([1, 10, 5])
    w = ipw_sample_weights(clicks, positions, eta=1.0, clip=100.0)
    assert w[1] > w[0]        # click at rank 10 counts more than at rank 1
    assert w[2] == 1.0        # non-clicks unweighted


def test_mmr_prefers_diverse_items():
    # items 0 and 1 are near-duplicates; item 2 is different but lower-scored
    emb = np.array([[1.0, 0.0], [0.999, 0.045], [0.0, 1.0]])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    scores = np.array([1.0, 0.95, 0.5])
    picked = mmr_rerank(scores, emb, k=2, lam=0.5)
    assert picked == [0, 2]   # skips the duplicate, takes the diverse item


def test_map_penalizes_missed_relevant_items():
    """The bug-catcher: finding 1-of-5 must score worse than 1-of-1."""
    from mind_ltr.ranking.metrics import map_at_k

    y_five = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    s_five = np.array([0.99, 0.01, 0.02, 0.03, 0.04, 0.9, 0.8, 0.7, 0.6, 0.5])
    y_one = np.array([1, 0, 0, 0, 0])
    s_one = np.array([0.99, 0.9, 0.8, 0.7, 0.6])

    assert map_at_k(y_one, s_one, k=5) == 1.0          # truly perfect
    assert map_at_k(y_five, s_five, k=5) == 1.0 / 5.0  # found 1 of 5


def test_map_known_value():
    """Hand-computed: relevant at ranks 1 and 3 of 4, both relevant items exist.

    AP@4 = (P@1 + P@3) / 2 = (1/1 + 2/3) / 2 = 5/6
    """
    from mind_ltr.ranking.metrics import map_at_k

    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1])
    assert abs(map_at_k(y, s, k=4) - 5.0 / 6.0) < 1e-9
