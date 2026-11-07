"""Synthetic impression-log generator.

Purpose: let the whole training pipeline run end-to-end BEFORE we download
MIND, and — more importantly — give us a sandbox where we KNOW the ground
truth, so we can see position bias with our own eyes.

The generative story (a standard "examination model" / position-based model):

  1. Each item has a TRUE relevance in [0, 1], driven by hidden features.
  2. Items were shown to the user in some historical order (by an old,
     mediocre production ranker).
  3. The user EXAMINES position i with probability propensity[i] ~ 1/i^eta
     (users look at rank 1 far more than rank 10).
  4. click = examined AND relevant.

So clicks = relevance x position bias. A naive model trained on raw clicks
partially learns "whatever the old ranker put on top", not true relevance.
That is the position-bias problem, and IPW (inverse propensity weighting)
is the standard fix — we implement it in mind_ltr.rerank.position_bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticData:
    X: np.ndarray  # (n_rows, n_features) item features
    clicks: np.ndarray  # (n_rows,) observed labels (biased!)
    true_relevance: np.ndarray  # (n_rows,) hidden ground truth in {0,1}
    positions: np.ndarray  # (n_rows,) historical display position, 1-indexed
    group_sizes: np.ndarray  # impressions' sizes, LightGBM style


def make_impressions(
    n_impressions: int = 2000,
    items_per_impression: int = 20,
    n_features: int = 8,
    position_bias_eta: float = 1.0,
    seed: int = 42,
) -> SyntheticData:
    rng = np.random.default_rng(seed)
    n_rows = n_impressions * items_per_impression

    X = rng.normal(size=(n_rows, n_features))

    # True relevance: driven by features 1..n-1. Feature 0 ("popularity")
    # is only WEAKLY relevant — this is deliberate, see below.
    w = rng.normal(size=n_features)
    w[0] = 0.2  # popularity barely matters for relevance
    logits = X @ w - 1.0
    p_rel = 1.0 / (1.0 + np.exp(-logits))
    true_relevance = (rng.random(n_rows) < p_rel).astype(int)

    # Legacy production ranker: ranks almost entirely by POPULARITY
    # (feature 0). This is the confound that makes position bias dangerous:
    # popular items get top positions -> get examined -> get clicked, so in
    # the click log, feature 0 LOOKS strongly predictive of clicks even
    # though it barely predicts relevance. A naive model inherits this
    # mistake; IPW largely undoes it.
    legacy_score = 3.0 * X[:, 0] + rng.normal(scale=0.5, size=n_rows)

    clicks = np.zeros(n_rows, dtype=int)
    positions = np.zeros(n_rows, dtype=int)
    group_sizes = np.full(n_impressions, items_per_impression)

    start = 0
    for _ in range(n_impressions):
        sl = slice(start, start + items_per_impression)
        order = np.argsort(legacy_score[sl])[::-1]  # legacy display order
        pos = np.empty(items_per_impression, dtype=int)
        pos[order] = np.arange(1, items_per_impression + 1)
        positions[sl] = pos

        # Examination probability decays with position: 1 / pos^eta
        propensity = 1.0 / (pos.astype(float) ** position_bias_eta)
        examined = rng.random(items_per_impression) < propensity
        clicks[sl] = (examined & (true_relevance[sl] == 1)).astype(int)
        start += items_per_impression

    return SyntheticData(X, clicks, true_relevance, positions, group_sizes)
