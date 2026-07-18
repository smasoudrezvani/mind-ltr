"""Ranking metrics implemented from scratch.

Vocabulary
----------
An "impression" (or "query group") is one ranking event: a user was shown a
list of items, and we know which ones they clicked. All metrics below are
computed per impression and then averaged across impressions.

y_true : relevance labels for the items in ONE impression (1 = clicked).
y_score: model scores for the same items. Higher score = ranked earlier.
"""

from __future__ import annotations

import numpy as np


def dcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Discounted Cumulative Gain.

    DCG@k = sum_{i=1..k} (2^rel_i - 1) / log2(i + 1)

    Intuition: a relevant item contributes its "gain" (2^rel - 1),
    discounted by how far down the list it appears. Position 1 gets
    divided by log2(2)=1 (no discount), position 2 by log2(3)≈1.58, etc.
    A click at rank 1 is worth ~1.0; the same click at rank 10 ~0.29.
    """
    order = np.argsort(y_score)[::-1][:k]          # indices of top-k by score
    gains = (2.0 ** y_true[order]) - 1.0
    discounts = np.log2(np.arange(2, len(order) + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """Normalized DCG: DCG of our ranking divided by the best possible DCG.

    The normalization (dividing by "ideal DCG") makes impressions with
    different numbers of clicks comparable: NDCG is always in [0, 1].
    """
    ideal = dcg_at_k(y_true, y_true, k)            # rank by true labels = best case
    if ideal == 0.0:
        return 0.0                                  # no relevant items -> undefined; convention: 0
    return dcg_at_k(y_true, y_score, k) / ideal


def mrr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean Reciprocal Rank (single impression -> reciprocal rank).

    1 / (rank of the FIRST relevant item). Cares only about how quickly
    the user hits something relevant — good proxy for "time to first click".
    """
    order = np.argsort(y_score)[::-1]
    ranked_labels = y_true[order]
    hits = np.flatnonzero(ranked_labels > 0)
    if len(hits) == 0:
        return 0.0
    return 1.0 / (hits[0] + 1)


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """Fraction of all relevant items that appear in the top-k.

    This is the metric for the CANDIDATE GENERATION stage: the retriever's
    only job is to not lose relevant items before the ranker sees them.
    """
    n_relevant = float(np.sum(y_true > 0))
    if n_relevant == 0:
        return 0.0
    order = np.argsort(y_score)[::-1][:k]
    return float(np.sum(y_true[order] > 0)) / n_relevant


def evaluate_groups(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_sizes: np.ndarray,
    k: int = 10,
) -> dict[str, float]:
    """Average metrics over many impressions.

    `group_sizes` is the LightGBM-style group array: e.g. [5, 8, 3] means the
    first 5 rows belong to impression 1, next 8 to impression 2, etc.
    """
    ndcgs, mrrs, recalls = [], [], []
    start = 0
    for size in group_sizes:
        yt = y_true[start : start + size]
        ys = y_score[start : start + size]
        ndcgs.append(ndcg_at_k(yt, ys, k))
        mrrs.append(mrr(yt, ys))
        recalls.append(recall_at_k(yt, ys, k))
        start += size
    return {
        f"ndcg@{k}": float(np.mean(ndcgs)),
        "mrr": float(np.mean(mrrs)),
        f"recall@{k}": float(np.mean(recalls)),
        "n_impressions": len(group_sizes),
    }

def map_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """Mean Average Precision at k.

    AP@k = sum_{i=1..k} (P(i) * rel_i) / min(k, #relevant)
    where P(i) is precision at rank i and rel_i is 1 if the item at rank i is relevant, else 0.
    """
    order = np.argsort(y_score)[::-1][:k]
    ranked_labels = y_true[order]
    relevant_indices = np.flatnonzero(ranked_labels > 0)
    if len(relevant_indices) == 0:
        return 0.0
    precisions = [(i + 1) / (idx + 1) for i, idx in enumerate(relevant_indices)]
    return float(np.sum(precisions)) / min(k, len(relevant_indices))
