"""Maximal Marginal Relevance (MMR) re-ranking for diversity.

After the ranker scores candidates, the top of the list is often redundant
(five near-identical articles about the same game). MMR greedily builds the
final list by trading off relevance against similarity to what's already
selected:

    MMR(item) = lambda * relevance(item)
                - (1 - lambda) * max_{s in selected} sim(item, s)

lambda = 1.0 -> pure relevance (no diversity)
lambda = 0.0 -> pure diversity (ignores the ranker)

This is the "re-ranking" box in the prediction pipeline (Fig 6.23 in the
ByteByteGo course): a cheap post-processing step over the top ~hundreds.
"""

from __future__ import annotations

import numpy as np


def mmr_rerank(
    scores: np.ndarray,  # (n,) ranker scores, higher = better
    item_embeddings: np.ndarray,  # (n, d) L2-normalized embeddings
    k: int = 10,
    lam: float = 0.7,
) -> list[int]:
    """Return indices of the k items selected by MMR, in display order."""
    n = len(scores)
    k = min(k, n)
    sim = item_embeddings @ item_embeddings.T  # cosine sim (normalized inputs)

    selected: list[int] = []
    candidates = set(range(n))
    while len(selected) < k:
        best_idx, best_val = -1, -np.inf
        for i in candidates:
            redundancy = max((sim[i, j] for j in selected), default=0.0)
            val = lam * scores[i] - (1.0 - lam) * redundancy
            if val > best_val:
                best_idx, best_val = i, val
        selected.append(best_idx)
        candidates.remove(best_idx)
    return selected
