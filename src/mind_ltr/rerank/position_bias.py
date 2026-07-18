"""Position-bias correction via Inverse Propensity Weighting (IPW).

The problem
-----------
Click logs are biased: users click items at rank 1 more often simply because
they SEE them more often. A model trained naively on clicks learns to imitate
the old ranker instead of learning true relevance.

The examination model
---------------------
P(click) = P(examined | position) * P(relevant | item, user)
                └── propensity ──┘

If we know the propensity of each position, we can reweight each clicked
example by 1/propensity: a click at rank 10 (rarely examined) counts MORE
than a click at rank 1, exactly undoing the bias. In LightGBM this is just
`sample_weight` — no change to the objective.

Estimating propensities
-----------------------
In industry: result randomization (swap ranks 1<->k for a small % of traffic)
or intervention-free estimators (e.g. regression EM, dual learning).
Here we provide:
  * `propensity_power_law` — the standard 1/pos^eta parametric assumption,
    which is what most papers use as a baseline and what our synthetic
    generator actually follows.
  * TODO(step 6): estimate eta from data by comparing CTR curves of item
    pairs that appeared at multiple positions (a mini "randomization" study
    is possible on MIND because the same news item shows up at many ranks).
"""

from __future__ import annotations

import numpy as np


def propensity_power_law(positions: np.ndarray, eta: float = 1.0) -> np.ndarray:
    """P(examined | position) = 1 / position^eta, positions are 1-indexed."""
    return 1.0 / (positions.astype(float) ** eta)


def ipw_sample_weights(
    clicks: np.ndarray,
    positions: np.ndarray,
    eta: float = 1.0,
    clip: float = 10.0,
) -> np.ndarray:
    """Per-row training weights.

    Clicked rows get weight 1/propensity (clipped, because tiny propensities
    give huge, high-variance weights — the classic IPW bias/variance
    trade-off). Non-clicked rows keep weight 1.0; treating unclicked items
    at low ranks as "true negatives" is itself an approximation, and a nice
    thing to mention in interviews (see: unbiased LTR literature).
    """
    prop = propensity_power_law(positions, eta)
    weights = np.ones_like(prop)
    clicked = clicks > 0
    weights[clicked] = np.minimum(1.0 / prop[clicked], clip)
    return weights
