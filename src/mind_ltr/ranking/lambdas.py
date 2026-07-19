"""LambdaRank/LambdaMART gradients ("lambdas") from scratch.

For every pair (i, j) in one impression where label_i > label_j:

    rho_ij     = sigma / (1 + exp(sigma * (s_i - s_j)))      # RankNet dial
    |dNDCG_ij| = |gain_i - gain_j| * |disc_i - disc_j| / IDCG
    lambda_ij  = rho_ij * |dNDCG_ij|                          # LambdaRank

    lambda_i += lambda_ij      (force UP on the more relevant item)
    lambda_j -= lambda_ij      (force DOWN on the less relevant item)

gain_i = 2^label_i - 1,  disc_i = 1 / log2(1 + rank_i)  with rank from the
CURRENT model scores. LambdaMART = feed these per-item lambdas (and their
second derivatives) to gradient-boosted trees as pseudo-residuals.

Everything below is vectorized with pairwise matrices per impression.
"""

from __future__ import annotations

import numpy as np


def _ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """1-indexed rank of each item under descending score order."""
    order = np.argsort(scores)[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def lambdas_one_group(
    labels: np.ndarray,
    scores: np.ndarray,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-item (lambda, hessian) for ONE impression.

    Sign convention: positive lambda = "score should increase".
    """
    n = len(labels)
    grad = np.zeros(n)
    hess = np.zeros(n)

    gains = (2.0 ** labels) - 1.0
    ideal = np.sort(gains)[::-1]
    idcg = float(np.sum(ideal / np.log2(np.arange(2, n + 2))))
    if idcg == 0.0:                      # no relevant items -> no signal
        return grad, hess

    discounts = 1.0 / np.log2(1.0 + _ranks_from_scores(scores))

    # Pairwise matrices; entry [i, j] describes the ordered pair (i, j).
    s_diff = scores[:, None] - scores[None, :]
    rho = sigma / (1.0 + np.exp(sigma * s_diff))                # RankNet dial
    dndcg = (
        np.abs(gains[:, None] - gains[None, :])
        * np.abs(discounts[:, None] - discounts[None, :])
        / idcg
    )
    pair_lambda = rho * dndcg
    # d(rho)/d(s_i) magnitude -> Newton hessian per pair (always positive):
    pair_hess = rho * (sigma - rho) * dndcg

    more_relevant = labels[:, None] > labels[None, :]           # mask [i, j]
    masked_lambda = pair_lambda * more_relevant
    masked_hess = pair_hess * more_relevant

    # Row sum: force UP on i from every pair (i, j) it should win.
    # Column sum: force DOWN on j from every pair (i, j) it should lose.
    grad = masked_lambda.sum(axis=1) - masked_lambda.sum(axis=0)
    hess = masked_hess.sum(axis=1) + masked_hess.sum(axis=0)
    return grad, hess


def lambdas_all_groups(
    labels: np.ndarray,
    scores: np.ndarray,
    group_sizes: np.ndarray,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenated (lambda, hessian) across impressions, LightGBM layout."""
    grad = np.zeros(len(labels))
    hess = np.zeros(len(labels))
    start = 0
    for size in group_sizes:
        sl = slice(start, start + int(size))
        grad[sl], hess[sl] = lambdas_one_group(labels[sl], scores[sl], sigma)
        start += int(size)
    return grad, hess


def make_lgb_objective(group_sizes: np.ndarray, sigma: float = 1.0):
    """Wrap our lambdas as a LightGBM custom objective.

    LightGBM MINIMIZES and expects (grad, hess) w.r.t. raw scores, where a
    NEGATIVE grad means "increase this score". Our lambda is an ascent
    force (positive = increase), so we return -lambda.
    """

    def objective(preds: np.ndarray, train_data) -> tuple[np.ndarray, np.ndarray]:
        labels = train_data.get_label()
        grad, hess = lambdas_all_groups(labels, preds, group_sizes, sigma)
        return -grad, np.maximum(hess, 1e-9)   # floor hess for stability

    return objective
